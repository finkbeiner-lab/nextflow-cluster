#!/usr/bin/env python3
"""Training harness for the 2D neurite-segmentation U-Net.

Pipeline: rasterized ``(image, mask)`` pairs -> :class:`NeuriteDataset` ->
:class:`UNet` -> ``SoftDiceClDiceBCELoss`` (BCE + soft-Dice + soft-clDice) ->
Adam. Each epoch logs training loss and validation skeleton-F1 (the same tol=3
distance-transform metric as the benchmark) and the best-F1 checkpoint is saved.

Train/val split is BY FIELD (well) to avoid leakage: crops of a field never
appear in both splits. With only two annotated fields today (C03_t1, I03_t1)
the "dry run" mode holds one field out for validation and overfits the other —
this cannot generalise, it only proves the loop runs end to end and that loss
decreases and an F1 number is produced.

    ***  REAL TRAINING NEEDS MORE DATA + GPU.  ***
    The current 2-field set is a scaffold. Track 0 owns extending the SNT
    annotation set; once there are enough fields, run full training on the
    cluster GPU (``--device cuda``, more epochs, larger ``--crop`` / model
    ``--depth``). The code here is unchanged for that; only the data and
    compute scale up.

Examples::

    # End-to-end CPU dry run on the two existing fields (a few epochs):
    python train.py --dry-run \
        --data ml/neurite/data --out ml/neurite/runs/dryrun

    # Real run (later, on GPU):
    python train.py --data <big_data_dir> --out runs/full \
        --device cuda --epochs 200 --crop 512 --depth 4 --base 32
"""

from __future__ import annotations

import argparse
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import NeuriteDataset, list_pairs
from losses import SoftDiceClDiceBCELoss
from metrics import best_threshold_f1
from model import build_model


def split_by_field(
    pairs: List[Tuple[str, str, str]], val_frac: float = 0.5
) -> Tuple[List, List]:
    """Split ``(img, mask, skel)`` triples into train/val by field.

    Args:
        pairs: All field triples.
        val_frac: Fraction of fields (rounded up, at least 1) held out for
            validation.

    Returns:
        ``(train_pairs, val_pairs)``. With a single field, that field is used
        for both so the loop can still run.
    """
    if len(pairs) == 1:
        return pairs, pairs
    n_val = max(1, int(round(len(pairs) * val_frac)))
    return pairs[:-n_val], pairs[-n_val:]


def split_by_stems(
    pairs: List[Tuple[str, str, str]], val_stems: List[str]
) -> Tuple[List, List]:
    """Split fields into train/val by explicit substring match on the img path.

    Lets the caller name the validation fields (e.g. ``["H15", "I07"]``) so the
    held-out set is a deliberate, balanced choice (one CTR + one XDP line, both
    lines still represented in training) rather than an alphabetical tail. Any
    field whose ``_img.npy`` path contains one of ``val_stems`` goes to val; the
    rest to train.

    Args:
        pairs: All field triples.
        val_stems: Substrings identifying validation fields.

    Returns:
        ``(train_pairs, val_pairs)``.

    Raises:
        SystemExit: If a stem matches nothing, or all/no fields land in val.
    """
    val, train = [], []
    for trip in pairs:
        if any(s in trip[0] for s in val_stems):
            val.append(trip)
        else:
            train.append(trip)
    for s in val_stems:
        if not any(s in trip[0] for trip in pairs):
            raise SystemExit(f"--val-stems '{s}' matched no field in the data dir")
    if not val or not train:
        raise SystemExit(
            f"--val-stems produced train={len(train)} val={len(val)}; need both non-empty"
        )
    return train, val


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    val_pairs: List[Tuple[str, str, str]],
    device: torch.device,
) -> Tuple[float, float]:
    """Run full-field inference on val fields and compute best-threshold F1.

    Args:
        model: Trained model (set to eval by caller or here).
        val_pairs: Validation field triples.
        device: Torch device.

    Returns:
        ``(mean_f1, mean_best_threshold)`` over validation fields.
    """
    from dataset import percentile_normalize
    from infer import predict_probmap

    model.eval()
    f1s, thrs = [], []
    for img_path, _, skel_path in val_pairs:
        image = percentile_normalize(np.load(img_path))
        gt = (
            np.load(skel_path).astype(bool)
            if skel_path
            else np.load(img_path.replace("_img", "_mask")).astype(bool)
        )
        prob = predict_probmap(model, image, device=device, normalize=False)
        t, f = best_threshold_f1(gt, prob)
        f1s.append(f)
        thrs.append(t)
    return float(np.mean(f1s)), float(np.mean(thrs))


def train(args: argparse.Namespace) -> None:
    """Run the training loop.

    Args:
        args: Parsed CLI arguments.
    """
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)

    pairs = list_pairs(args.data)
    if not pairs:
        raise SystemExit(
            f"no rasterized pairs in {args.data}; run rasterize_traces.py first"
        )
    if args.val_stems:
        val_stems = [s.strip() for s in args.val_stems.split(",") if s.strip()]
        train_pairs, val_pairs = split_by_stems(pairs, val_stems)
    else:
        train_pairs, val_pairs = split_by_field(pairs, args.val_frac)
    print(f"fields: {len(pairs)} total | train={len(train_pairs)} val={len(val_pairs)}")
    for _, p in enumerate(val_pairs):
        print(f"  val field: {os.path.basename(p[0])[:-len('_img.npy')]}")

    train_ds = NeuriteDataset(
        train_pairs, crop=args.crop, length=args.iters_per_epoch,
        train=True, seed=args.seed,
    )
    loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0)

    model = build_model(depth=args.depth, base_channels=args.base).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model: UNet depth={args.depth} base={args.base} params={n_params:,} "
          f"device={device}")

    loss_fn = SoftDiceClDiceBCELoss(
        iters=args.cldice_iters, pos_weight=args.pos_weight
    )
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_f1 = -1.0
    best_path = os.path.join(args.out, "best.pt")
    print(f"{'epoch':>5} {'train_loss':>11} {'val_F1':>8} {'thr':>5}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for image, mask in loader:
            image, mask = image.to(device), mask.to(device)
            opt.zero_grad()
            logits = model(image)
            loss = loss_fn(logits, mask)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        val_f1, val_thr = evaluate(model, val_pairs, device)
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "depth": args.depth,
                        "base_channels": args.base,
                        "in_channels": 1,
                    },
                    "epoch": epoch,
                    "val_f1": val_f1,
                    "threshold": val_thr,
                },
                best_path,
            )
            flag = "  *best"
        print(f"{epoch:5d} {train_loss:11.4f} {val_f1:8.3f} {val_thr:5.2f}{flag}")

    print(f"\nbest val F1 = {best_f1:.3f}  ->  {best_path}")
    if args.dry_run:
        print(
            "\nDRY RUN complete. This overfits a tiny set to prove the harness "
            "runs; it does NOT generalise. Real training needs Track 0's larger "
            "SNT annotation set + GPU."
        )


def build_argparser() -> argparse.ArgumentParser:
    """Build the CLI parser.

    Returns:
        Configured :class:`argparse.ArgumentParser`.
    """
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="ml/neurite/data", help="rasterized pairs dir")
    ap.add_argument("--out", default="ml/neurite/runs/dryrun", help="output dir")
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--crop", type=int, default=256)
    ap.add_argument("--iters-per-epoch", type=int, default=40,
                    help="random crops sampled per epoch")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--depth", type=int, default=4)
    ap.add_argument("--base", type=int, default=16, help="base channel count")
    ap.add_argument("--cldice-iters", type=int, default=10,
                    help="soft-skeletonization iterations in clDice")
    ap.add_argument("--pos-weight", type=float, default=5.0,
                    help="BCE positive-class weight (thin structures are rare)")
    ap.add_argument("--val-frac", type=float, default=0.5)
    ap.add_argument("--val-stems", default="",
                    help="comma-separated substrings naming val fields "
                         "(e.g. 'H15,I07'); overrides --val-frac when set")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="tiny CPU config to prove the loop runs on the 2 existing fields",
    )
    return ap


def apply_dry_run_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """Shrink the config for a fast CPU dry run.

    Args:
        args: Parsed args.

    Returns:
        Possibly-modified args (only when ``--dry-run`` and the user left the
        heavy defaults in place).
    """
    if args.dry_run:
        args.device = "cpu"
        # Small, fast, CPU-friendly settings unless the user overrode them.
        args.epochs = min(args.epochs, 8)
        args.crop = min(args.crop, 128)
        args.depth = min(args.depth, 3)
        args.base = min(args.base, 8)
        args.iters_per_epoch = min(args.iters_per_epoch, 16)
        args.batch = min(args.batch, 2)
    return args


if __name__ == "__main__":
    parsed = apply_dry_run_defaults(build_argparser().parse_args())
    train(parsed)
