#!/usr/bin/env python
"""Click-to-add cell annotation (SNT-style) for Cellpose fine-tuning, powered by micro-SAM.

Opens each crop with the model's current cells as outlines over the GFP image. You only add
what the model missed: click a dim cell and SAM draws its boundary; click further along a
tail to extend it. Everything autosaves to <out_dir>/<tag>_masks.tif (uint16 labels), which
are the training masks (model cells you didn't touch + the cells you added).

Mouse (in pan/zoom mode -- drag still pans):
    click            add the cell under the cursor (SAM; existing cells are kept as-is)
    cmd-click        add another point to the LAST cell (extend a tail / grow it)
    option-click     mark "not part of the last cell" (shrink it)
    shift-click      delete the cell under the cursor (junk / bad model cell)
Keys:
    c   cycle SAM's 3 size guesses for the last cell     z   undo last change
    n   save + next crop     p   previous crop     h   hide/show outlines

Usage:
    python click_annotate.py <export_dir> <out_dir> [--model vit_b_lm] [--start TAG]
"""
import argparse
import json
import os
import sys

import napari
import numpy as np
import pandas as pd
import tifffile
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_fiji_set import DEFAULT_OVERRIDES, PRIORITY  # noqa: E402  same prefill + order as Fiji set

NEG_RADIUS = 120     # existing cells whose centroid is this close become negative prompts
MIN_AREA = 150       # ignore SAM masks smaller than this (px)
MAX_NEG = 6          # at most this many neighbor negatives (too many confuses SAM)


def display_uint8(raw: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Gamma-0.5 stretch -> uint8 RGB. Lifts dim cytoplasm so SAM (and you) can see it."""
    g = np.clip((raw - lo) / (hi - lo + 1e-9), 0, 1) ** 0.5
    return (g * 255).astype(np.uint8)


class Session:
    def __init__(self, export_dir: str, out_dir: str, model_type: str, start: str | None):
        from micro_sam import util
        self.util = util
        self.export_dir, self.out_dir = export_dir, out_dir
        os.makedirs(os.path.join(out_dir, "embeddings"), exist_ok=True)
        man = pd.read_csv(os.path.join(export_dir, "manifest.csv"))
        rank = {t: i for i, t in enumerate(PRIORITY)}
        man = man.assign(order=[rank.get(t, len(PRIORITY) + i) for i, t in enumerate(man.tag)])
        self.man = man.sort_values("order").reset_index(drop=True)
        self.i = int(self.man.index[self.man.tag == start][0]) if start else 0
        print(f"loading SAM model {model_type} (first run downloads weights) ...", flush=True)
        self.predictor = util.get_sam_model(model_type=model_type)
        self.viewer = napari.Viewer(title="click_annotate")
        self.history: list[np.ndarray] = []
        self.last = None   # dict(id, pos=[...], neg=[...], cands, k) for the most recent cell
        self._bind()
        self.load()

    # ---------- data ----------
    @property
    def tag(self) -> str:
        return self.man.tag[self.i]

    def mask_path(self) -> str:
        return os.path.join(self.out_dir, f"{self.tag}_masks.tif")

    def load(self) -> None:
        r = self.man.iloc[self.i]
        raw = tifffile.imread(os.path.join(self.export_dir, "crops", f"{r.tag}.tif")).astype(np.float32)
        self.img8 = display_uint8(raw, r.lo, r.hi)
        if os.path.exists(self.mask_path()):
            cells = tifffile.imread(self.mask_path()).astype(np.uint16)
            src = "your saved labels"
        else:
            v = DEFAULT_OVERRIDES.get(r.tag, "d50")
            cells = np.load(os.path.join(self.export_dir, "preds", f"{r.tag}.npz"))[v].astype(np.uint16)
            src = f"model ({v})"
        self.n_start = int(len(np.unique(cells)) - 1)
        self.history, self.last = [], None
        self.viewer.layers.clear()
        self.viewer.add_image(self.img8, name="GFP16 (gamma 0.5)")
        self.cells = self.viewer.add_labels(cells, name="cells", opacity=0.9)
        self.cells.contour = 2
        self.cells.mode = "pan_zoom"
        print(f"[{self.i + 1}/{len(self.man)}] {r.tag} ({r.line} {r.dose} {r.tp}, {r.kind}, {r.split}): "
              f"{self.n_start} cells from {src}. Computing SAM embedding ...", flush=True)
        emb_path = os.path.join(self.out_dir, "embeddings", f"{r.tag}.zarr")
        self.emb = self.util.precompute_image_embeddings(self.predictor, self.img8, save_path=emb_path)
        self.util.set_precomputed(self.predictor, self.emb)
        self.viewer.reset_view()
        self.status("ready -- click dim cells to add them")

    def save(self) -> None:
        tifffile.imwrite(self.mask_path(), self.cells.data.astype(np.uint16))
        n = int(len(np.unique(self.cells.data)) - 1)
        with open(os.path.join(self.out_dir, f"{self.tag}_log.json"), "w") as f:
            json.dump(dict(tag=self.tag, n_model_start=self.n_start, n_now=n), f)

    def status(self, msg: str) -> None:
        n = int(len(np.unique(self.cells.data)) - 1)
        self.viewer.status = msg
        self.viewer.title = (f"[{self.i + 1}/{len(self.man)}] {self.tag} | cells {n} "
                             f"(model {self.n_start}, net {n - self.n_start:+d}) | {msg}")

    # ---------- SAM ----------
    def predict(self, pos: list, neg: list) -> list:
        """SAM's 3 multimask candidates for these points, restricted to unlabeled pixels."""
        pts = np.array(pos + neg, dtype=np.float32)[:, ::-1]          # (row,col) -> (x,y)
        lbl = np.array([1] * len(pos) + [0] * len(neg))
        masks, scores, _ = self.predictor.predict(point_coords=pts, point_labels=lbl, multimask_output=True)
        order = np.argsort(-scores)
        cands = []
        free = (self.cells.data == 0) | (self.cells.data == (self.last["id"] if self.last else -1))
        for j in order:
            m = masks[j] & free
            lab, _ = ndimage.label(m)
            hit = {int(lab[int(r), int(c)]) for r, c in pos} - {0}   # pieces touching any + point
            keep = np.isin(lab, list(hit)) if hit else np.zeros_like(m)
            if keep.sum() >= MIN_AREA:
                cands.append(keep)
        return cands

    def negatives_near(self, r: int, c: int) -> list:
        data = self.cells.data
        ids = np.unique(data[max(0, r - NEG_RADIUS):r + NEG_RADIUS, max(0, c - NEG_RADIUS):c + NEG_RADIUS])
        ids = ids[ids > 0]
        if not len(ids):
            return []
        cents = np.array(ndimage.center_of_mass(data > 0, data, ids))
        near = np.argsort(np.hypot(cents[:, 0] - r, cents[:, 1] - c))[:MAX_NEG]
        return [list(map(float, cents[j])) for j in near]

    def apply_last(self) -> None:
        d = self.cells.data
        d[d == self.last["id"]] = 0
        if self.last["cands"]:
            d[self.last["cands"][self.last["k"] % len(self.last["cands"])]] = self.last["id"]
        self.cells.refresh()
        self.save()

    # ---------- interaction ----------
    def click(self, pos: tuple, mods: set) -> None:
        r, c = int(round(pos[0])), int(round(pos[1]))
        H, W = self.cells.data.shape
        if not (0 <= r < H and 0 <= c < W):
            return
        under = int(self.cells.data[r, c])
        if "Shift" in mods:
            if under:
                self.history.append(self.cells.data.copy())
                self.cells.data[self.cells.data == under] = 0
                self.cells.refresh(); self.save(); self.last = None
                self.status(f"deleted cell {under}")
            return
        if ("Meta" in mods or "Control" in mods or "Alt" in mods) and self.last:
            self.history.append(self.cells.data.copy())
            (self.last["neg"] if "Alt" in mods else self.last["pos"]).append([r, c])
            self.last["cands"] = self.predict(self.last["pos"], self.last["neg"]); self.last["k"] = 0
            self.apply_last()
            self.status(f"cell {self.last['id']}: {len(self.last['pos'])} + / {len(self.last['neg'])} - points")
            return
        if under:
            self.status(f"that's cell {under} already (shift-click deletes it)")
            return
        self.history.append(self.cells.data.copy())
        new_id = int(self.cells.data.max()) + 1
        self.last = dict(id=new_id, pos=[[r, c]], neg=self.negatives_near(r, c), cands=[], k=0)
        self.last["cands"] = self.predict(self.last["pos"], self.last["neg"])
        if not self.last["cands"]:
            self.history.pop(); self.last = None
            self.status("SAM found nothing there -- try clicking the brightest part of the cell")
            return
        self.apply_last()
        self.status(f"added cell {new_id}  (c = other size, cmd-click = extend, z = undo)")

    def _bind(self) -> None:
        v = self.viewer

        @v.mouse_drag_callbacks.append
        def _on_mouse(viewer, event):
            if event.button != 1:
                return
            start = np.array(event.position)
            mods = {str(m) for m in event.modifiers}
            moved = False
            yield
            while event.type == "mouse_move":
                if np.linalg.norm(np.array(event.position) - start) > 3:
                    moved = True
                yield
            if not moved and self.cells.mode == "pan_zoom":
                self.click(tuple(self.cells.world_to_data(start)), mods)

        @v.bind_key("c", overwrite=True)
        def _cycle(viewer):
            if self.last and len(self.last["cands"]) > 1:
                self.last["k"] += 1; self.apply_last()
                self.status(f"cell {self.last['id']}: guess {self.last['k'] % len(self.last['cands']) + 1}"
                            f"/{len(self.last['cands'])}")

        @v.bind_key("z", overwrite=True)
        def _undo(viewer):
            if self.history:
                self.cells.data = self.history.pop(); self.last = None
                self.save(); self.status("undone")

        @v.bind_key("n", overwrite=True)
        def _next(viewer):
            self.save()
            if self.i + 1 < len(self.man):
                self.i += 1; self.load()
            else:
                self.status("last crop -- all saved. Tell Claude you're done.")

        @v.bind_key("p", overwrite=True)
        def _prev(viewer):
            self.save()
            if self.i > 0:
                self.i -= 1; self.load()

        @v.bind_key("h", overwrite=True)
        def _hide(viewer):
            self.cells.visible = not self.cells.visible


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("export_dir"); ap.add_argument("out_dir")
    ap.add_argument("--model", default="vit_b_lm")
    ap.add_argument("--start", default=None)
    a = ap.parse_args()
    Session(a.export_dir, a.out_dir, a.model, a.start)
    napari.run()


if __name__ == "__main__":
    main()
