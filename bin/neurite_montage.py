#!/usr/bin/env python
"""Whole-well montage per-cell neurite quantification with DB writes.

Production montage-level neurite module (the ``*_montage.py`` sibling of
``segmentation_montage.py`` / ``tracking_montage.py`` / ``overlay_montage.py``).
Unlike the tile-level ``bin/neurite.py``, this runs on the WHOLE-WELL montage so
neurites crossing tile seams are traced continuously and attributed to the soma
they physically connect to across the whole field.

Per well x timepoint it:

1. **Montage build** — stitches the morphology tiles into a whole-well montage
   with per-tile percentile normalization (+ optional rolling-background
   flattening + empty-tile guard). This matches how the clDice model was trained
   and removes tile-to-tile vignetting seams.
2. **Soma segmentation — Cellpose-SAM (default)** — per tile at the 2048px scale
   it was tuned at, labels offset to be globally unique in the montage, then the
   post-segmentation debris filter (MAD background cut + size-dependent floor).
3. **Neurites** — clDice U-Net over the full montage (tiled inference) +
   interior-seam suppression + small-object removal.
4. **Per-cell attribution + measurement** — geodesic soma-rooted ownership, then
   the per-cell neurite metrics (total length, branch/end points, primary
   neurites, max branch length, skeleton px).
5. **DB write** — ``celldata`` (the Cellpose somas, keyed by
   ``randomcellid_montage`` + montage centroid) AND ``neuritecelldata`` (keyed to
   those cells). Montage cells are written to ONE representative tiledata row per
   (well, timepoint); idempotency deletes only montage-tagged cells
   (``randomcellid_montage`` not null), so any pre-existing per-tile ``celldata``
   is preserved.

The neurite compute reuses the audited, validated pipeline in
``ml/neurite/percell_integrate.py`` (geodesic ``attribute``, ``per_soma_lengths``,
tiled ``predict_tiled``, ``segment_somas``) via ``ml/neurite/montage_percell.py``,
and the per-cell skeleton-graph metrics in ``bin/neurite.py``; the DB conventions
follow ``bin/segmentation_helper_montage.py``. ``torch``/``cellpose`` are imported
lazily so this file imports on a torch-less machine.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import uuid
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

logger = logging.getLogger("neurite_montage")

# The montage neurite compute lives in the offline training package (ml/neurite),
# baked at /ml in the container and pointed at by NEURITE_ML_DIR. Add it to
# sys.path so `montage_percell` / `percell_integrate` import (mirrors
# bin/neurite_model.py). Harmless at import time — those modules (torch/cellpose)
# are only imported lazily inside run()/measure_montage_neurites.
_ML_DIR = os.environ.get(
    "NEURITE_ML_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "ml", "neurite"),
)
if _ML_DIR not in sys.path:
    sys.path.insert(0, _ML_DIR)


@dataclass
class SomaNeurite:
    """Per-soma montage measurement (celldata shape props + neurite metrics)."""

    label: int                       # montage soma label (-> randomcellid_montage)
    centroid_y: float                # montage-coordinate weighted centroid (row)
    centroid_x: float                # montage-coordinate weighted centroid (col)
    area_px: int                     # soma area (px)
    perimeter: float
    solidity: float
    extent: float
    eccentricity: float
    axis_major_length: float
    axis_minor_length: float
    total_neurite_length: float      # summed geodesic skeleton length (px)
    n_branch_points: int             # skeleton nodes with degree >= 3
    n_end_points: int                # skeleton nodes with degree == 1
    n_primary_neurites: int          # branches leaving the soma
    max_branch_length: float         # longest path from the soma (px)
    n_skeleton_px: int               # attributed skeleton pixel count


# --------------------------------------------------------------------------- #
#  Compute core — montage neurite detection + per-soma metrics
# --------------------------------------------------------------------------- #
def measure_montage_neurites(
    image01: np.ndarray,
    soma_labels: np.ndarray,
    opt: argparse.Namespace,
) -> List[SomaNeurite]:
    """Detect neurites on a montage and measure them per soma.

    Args:
        image01: Whole-well montage, float32 in [0, 1], already per-tile
            normalized (the detector is called with ``normalize=False``).
        soma_labels: Montage soma-label image (int); globally-unique labels
            (0 = background), each label the cell's ``randomcellid_montage``.
        opt: Parsed args carrying detector + tuning params.

    Returns:
        One :class:`SomaNeurite` per soma label present in ``soma_labels``.
    """
    from skimage.measure import regionprops
    from skimage.morphology import remove_small_objects

    from percell_integrate import attribute, per_soma_lengths
    from neurite import (_skeleton_graph_metrics, _count_primary_neurites,
                         _max_path_from_soma)

    grid = int(opt.neurite_grid)

    # 1. Neurite probability map over the whole montage.
    if opt.detector == "cldice":
        from neurite_model import predict_neurite_probmap
        prob = predict_neurite_probmap(
            image01, checkpoint=opt.checkpoint, device=opt.device,
            normalize=False, tile=opt.neurite_tile)   # image01 is pre-normalized
    else:
        from neurite import enhance_vesselness
        prob = enhance_vesselness(image01, opt.vesselness_sigma_min,
                                  opt.vesselness_sigma_max, opt.vesselness_sigma_steps)
    raw_mask = prob >= opt.neurite_prob_threshold

    # 2. Interior-seam suppression: per-tile normalization leaves a straight
    #    intensity step at each interior tile boundary that the ridge-sensitive
    #    U-Net fires on. Zero a thin band at each interior seam; a real neurite
    #    crossing a seam loses only ~2*band px (negligible for length).
    band = int(getattr(opt, "neurite_seam_band", 0) or 0)
    if band > 0 and grid > 1:
        th_m, tw_m = image01.shape[0] // grid, image01.shape[1] // grid
        for kk in range(1, grid):
            raw_mask[kk * th_m - band:kk * th_m + band, :] = False
            raw_mask[:, kk * tw_m - band:kk * tw_m + band] = False

    neurite_mask = remove_small_objects(
        raw_mask, int(getattr(opt, "neurite_min_object", 25)))

    # 3. Geodesic soma-rooted attribution (validated): each foreground pixel is
    #    owned by the soma it connects to across the whole montage.
    owner, skel = attribute(soma_labels, neurite_mask)
    lengths = per_soma_lengths(owner, skel, soma_labels)   # {label: (len, skel_px)}
    props = regionprops(soma_labels, intensity_image=image01)

    # Per-cell metrics run on a CROP, not the whole montage. find_objects gives
    # each owner label's bounding box (the soma + its attributed neurites) in a
    # single pass, so the per-cell graph metrics cost O(sum of cell bboxes)
    # instead of O(n_cells * montage): on a dense 8192^2 well that is the
    # difference between ~1 min and ~50 min. The crop always contains every
    # ``owner == lab`` pixel (that is what the bbox bounds) plus a small pad for
    # the soma-dilation ring, so the metrics are identical to the whole-image
    # computation — just translated into a smaller array.
    from scipy import ndimage as ndi
    owner_slices = ndi.find_objects(owner)
    pad = int(opt.soma_dilation) + 2
    H, W = owner.shape

    results: List[SomaNeurite] = []
    for p in props:
        lab = int(p.label)
        length, skpx = lengths.get(lab, (0.0, 0))
        sl = owner_slices[lab - 1] if lab - 1 < len(owner_slices) else None
        if skpx > 0 and sl is not None:
            ys, xs = sl
            ysl = slice(max(0, ys.start - pad), min(H, ys.stop + pad))
            xsl = slice(max(0, xs.start - pad), min(W, xs.stop + pad))
            cell_skel = (owner[ysl, xsl] == lab) & skel[ysl, xsl]
            soma_c = soma_labels[ysl, xsl] == lab
            n_branch, n_end = _skeleton_graph_metrics(cell_skel)
            n_primary = _count_primary_neurites(cell_skel, soma_c, int(opt.soma_dilation))
            max_len = _max_path_from_soma(cell_skel, soma_c)
        else:
            n_branch = n_end = n_primary = 0
            max_len = 0.0
        cy, cx = p.centroid_weighted
        results.append(SomaNeurite(
            label=lab, centroid_y=float(cy), centroid_x=float(cx),
            area_px=int(p.area), perimeter=float(p.perimeter),
            solidity=float(p.solidity), extent=float(p.extent),
            eccentricity=float(p.eccentricity),
            axis_major_length=float(p.axis_major_length),
            axis_minor_length=float(p.axis_minor_length),
            total_neurite_length=round(float(length), 3),
            n_branch_points=int(n_branch), n_end_points=int(n_end),
            n_primary_neurites=int(n_primary),
            max_branch_length=round(float(max_len), 3),
            n_skeleton_px=int(skpx)))
    return results


# --------------------------------------------------------------------------- #
#  Pipeline orchestration — montage build + DB writes
# --------------------------------------------------------------------------- #
class NeuriteMontage:
    """Runs montage neurite quantification over an experiment, one well at a time."""

    def __init__(self, opt: argparse.Namespace) -> None:
        from normalization import Normalize
        self.opt = opt
        self.Norm = Normalize(opt)
        self._cp_model = None

    def _cellpose(self):
        """Lazily construct the Cellpose-SAM model (cached).

        ``use_bfloat16=False`` is critical on V100/Volta GPUs. cellpose defaults
        to bfloat16 and casts the whole ViT-L forward to bf16 with no autocast
        and no GPU-arch check; Volta (sm_70) has no native bf16, so the forward
        is emulated and runs ~10x slower (~4 min/well -> ~50 min/well). fp32 is
        GPU-arch-agnostic and fast (cellpose tiles internally, so memory stays
        bounded). ``pretrained_model='cpsam'`` pins the checkpoint the soma
        recipe was validated with (cpsam_v2 is a newer same-architecture
        checkpoint that became the default after a cellpose version bump).
        """
        if self._cp_model is None:
            from cellpose import models
            self._cp_model = models.CellposeModel(
                gpu=(self.opt.device == "cuda"),
                pretrained_model="cpsam",
                use_bfloat16=False,
            )
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Cellpose-SAM on GPU %s (fp32, bf16 disabled)",
                                torch.cuda.get_device_name(0))
                else:
                    logger.warning("Cellpose-SAM running on CPU (no CUDA visible) "
                                   "— this is ~10x+ slower; check --nv/--gres.")
            except Exception:  # noqa: BLE001
                pass
        return self._cp_model

    def run(self) -> None:
        import pandas as pd
        from sql import Database
        from montage_percell import build_montages

        Db = Database()
        df_all = self.Norm.get_df_for_training(['channeldata'])
        morph = df_all[df_all.channel == self.opt.morphology_channel]
        if self.opt.tile:  # 0 == all tiles
            morph = morph[morph.tile == self.opt.tile]
        if morph.empty:
            logger.warning('No tiles for morphology channel %s', self.opt.morphology_channel)
            print('Done.')
            return

        # The montage cell identity column lives in the live (reflected) schema.
        # Refuse to run if it is absent rather than fall back to randomcellid,
        # which would let idempotency deletes clobber per-tile celldata.
        celldata_cols = set(Db.meta.tables['celldata'].c.keys())
        if 'randomcellid_montage' not in celldata_cols:
            raise RuntimeError(
                "celldata has no 'randomcellid_montage' column in the live schema; "
                "montage neurite writes need it to stay disjoint from per-tile cells.")

        grid = int(self.opt.neurite_grid)
        n_expected = grid * grid
        cp = self._cellpose()
        total_cells = total_neur = 0

        for (well, timepoint), df in morph.groupby(['well', 'timepoint']):
            df = df.sort_values('tile')
            paths = [p for p in df.filename.tolist()]
            sites = [int(t) for t in df.tile.tolist()]
            if len(paths) != n_expected:
                logger.warning('%s T%s: found %d morphology tiles (need %d) — skipping',
                               well, timepoint, len(paths), n_expected)
                continue
            if any(p is None or not os.path.exists(p) for p in paths):
                logger.warning('%s T%s: missing raw tile file(s) — skipping', well, timepoint)
                continue

            rep = df.iloc[0]  # representative tiledata row for the montage cells
            morph_uuid = Db.get_table_uuid(
                'channeldata', dict(channel=self.opt.morphology_channel,
                                    welldata_id=rep.welldata_id))
            try:
                logger.info('%s T%s: building montage + Cellpose-SAM somas over %d tiles...',
                            well, timepoint, len(paths))
                image01, soma = build_montages(
                    cp, paths, sites, self.opt.soma_diameter, self.opt.soma_flow,
                    self.opt.soma_cellprob, self.opt.soma_clean_k,
                    flatten_size=self.opt.neurite_flatten_size)
                logger.info('%s T%s: %d somas segmented; tracing neurites on the montage...',
                            well, timepoint, int(soma.max()))
                measurements = measure_montage_neurites(image01, soma, self.opt)
            except Exception as exc:  # noqa: BLE001 - one bad well must not kill the run
                logger.warning('%s T%s FAILED: %s: %s', well, timepoint,
                               type(exc).__name__, exc)
                continue
            if not measurements:
                logger.info('%s T%s: no somas segmented', well, timepoint)
                continue

            # Idempotency: drop only prior montage cells on this rep row (+cascade
            # to their neuritecelldata); leave per-tile celldata untouched.
            self._delete_prior_montage_cells(Db, rep.tiledata_id)

            celldata_dcts: List[dict] = []
            neurite_dcts: List[dict] = []
            for m in measurements:
                cid = uuid.uuid4()
                celldata_dcts.append(dict(
                    id=cid,
                    experimentdata_id=rep.experimentdata_id,
                    welldata_id=rep.welldata_id,
                    tiledata_id=rep.tiledata_id,
                    randomcellid_montage=m.label,
                    centroid_x=m.centroid_x,
                    centroid_y=m.centroid_y,
                    area=float(m.area_px),
                    perimeter=m.perimeter,
                    solidity=m.solidity,
                    extent=m.extent,
                    eccentricity=m.eccentricity,
                    axis_major_length=m.axis_major_length,
                    axis_minor_length=m.axis_minor_length,
                ))
                neurite_dcts.append(dict(
                    experimentdata_id=rep.experimentdata_id,
                    welldata_id=rep.welldata_id,
                    tiledata_id=rep.tiledata_id,
                    celldata_id=cid,
                    channeldata_id=morph_uuid,
                    total_neurite_length=m.total_neurite_length,
                    n_branch_points=m.n_branch_points,
                    n_end_points=m.n_end_points,
                    n_primary_neurites=m.n_primary_neurites,
                    max_branch_length=m.max_branch_length,
                    n_skeleton_px=m.n_skeleton_px,
                ))

            Db.add_row(tablename='celldata', dct=celldata_dcts)
            Db.add_row(tablename='neuritecelldata', dct=neurite_dcts)
            n_with = sum(1 for m in measurements if m.n_skeleton_px > 0)
            total_cells += len(celldata_dcts)
            total_neur += n_with
            logger.info('%s T%s: %d somas (%d with neurites) written',
                        well, timepoint, len(celldata_dcts), n_with)
            print(f'{well} T{timepoint}: {len(celldata_dcts)} somas, '
                  f'{n_with} with neurites')

        logger.info('NEURITE_MONTAGE done: %d cells, %d with neurites',
                    total_cells, total_neur)
        print(f'Done. {total_cells} cells written, {total_neur} with neurites.')

    def _delete_prior_montage_cells(self, Db, tiledata_id) -> None:
        """Delete only montage-tagged celldata (randomcellid_montage not null)
        for this tiledata row, cascading to their neuritecelldata; per-tile cells
        (randomcellid_montage null) are left in place."""
        prior = Db.get_df_from_query('celldata', dict(tiledata_id=tiledata_id))
        if prior.empty or 'randomcellid_montage' not in prior.columns:
            return
        montage_rows = prior[prior['randomcellid_montage'].notna()]
        for cid in montage_rows['id'].tolist():
            Db.delete_based_on_duplicate_name(tablename='celldata', kwargs=dict(id=cid))


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse CLI args (experiment selection + detector + montage/soma params)."""
    ap = argparse.ArgumentParser(description="Montage per-cell neurite quantification (DB).")
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--morphology_channel", required=True)
    ap.add_argument("--chosen_wells", default="all")
    ap.add_argument("--wells_toggle", default="include")
    ap.add_argument("--chosen_timepoints", default="all")
    ap.add_argument("--timepoints_toggle", default="include")
    ap.add_argument("--chosen_channels", default="all")
    ap.add_argument("--channels_toggle", default="include")
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--img_norm_name", default="identity")
    # neurite detector
    ap.add_argument("--detector", choices=["frangi", "cldice"], default="cldice")
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--neurite_prob_threshold", type=float, default=0.5)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--neurite_tile", type=int, default=1024)
    ap.add_argument("--neurite_min_object", type=int, default=25)
    ap.add_argument("--neurite_seam_band", type=int, default=6)
    # montage build
    ap.add_argument("--neurite_grid", type=int, default=4)
    ap.add_argument("--neurite_flatten_size", type=int, default=128)
    # cellpose-SAM soma recipe (locked defaults)
    ap.add_argument("--soma_diameter", type=int, default=25)
    ap.add_argument("--soma_flow", type=float, default=0.6)
    ap.add_argument("--soma_cellprob", type=float, default=-1.0)
    ap.add_argument("--soma_clean_k", type=float, default=2.0)
    ap.add_argument("--soma_dilation", type=int, default=3)
    # classical Frangi params (only used when --detector frangi)
    ap.add_argument("--vesselness_sigma_min", type=float, default=1.0)
    ap.add_argument("--vesselness_sigma_max", type=float, default=8.0)
    ap.add_argument("--vesselness_sigma_steps", type=int, default=5)
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    print(args)
    logger.warning('Running NeuriteMontage from Database.')
    NeuriteMontage(args).run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
