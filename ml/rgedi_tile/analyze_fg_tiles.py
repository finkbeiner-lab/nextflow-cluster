#!/usr/bin/env python
"""Summarize the per-tile foreground readout (fg_tile_readout.py tiles.csv) per well and condition.

Usage:
    python analyze_fg_tiles.py <tiles.csv> <out_dir> [--exp hevo-pmsG-1]

Per well x timepoint the 9 tiles are summarized as mean / SD / SEM / IQR of the tile values, plus a
pixel-weighted pooled ratio. Ratios: median-based = median RFP / median GFP over foreground pixels;
mean-based = mean RFP / mean GFP; bg-sub = (median RFP - bg median RFP) / (median GFP - bg median GFP).
Fold change = well value / the same well's T0 value. Time axis = nominal 4 h spacing.

Writes well_tp.csv, condition_tp.csv and figures (PNG) to <out_dir>.
"""
import argparse
import os

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_plot, _fill = plt.Axes.plot, plt.Axes.fill_between
plt.Axes.plot = lambda ax, *a, **k: _plot(ax, *[np.asarray(x) if isinstance(x, pd.Series) else x for x in a], **k)
plt.Axes.fill_between = lambda ax, *a, **k: _fill(ax, *[np.asarray(x) if isinstance(x, pd.Series) else x for x in a], **k)

LAYOUT = {
    "hevo-pmsG-1": dict(line={"C": "TP0357", "D": "TP0359", "E": "TP0388", "F": "TP0398"},
                        dose_ms={3: 500, 4: 1000, 5: 5000, 6: 10000}, hours_per_tp=4),
    "hevo-pmsG-2": dict(line={"C": "TP057", "D": "TP057", "E": "TP059", "F": "TP059"},
                        dose_ms={3: 250, 4: 500, 5: 1000, 6: 2000, 7: 5000, 8: 0}, hours_per_tp=4),
}
# ordinal dose ramp (one hue, light -> dark); 0 ms control is neutral gray
BLUES = ["#86b6ef", "#5598e7", "#3987e5", "#256abf", "#1c5cab", "#0d366b"]
GRAY = "#8a8984"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
METRICS = {
    "ratio_median": "RFP / GFP (median-based)",
    "ratio_mean": "RFP / GFP (mean-based)",
    "ratio_bgsub": "RFP / GFP (background-subtracted)",
}


def dose_label(ms):
    return "0 ms (unstim)" if ms == 0 else (f"{ms // 1000} s" if ms >= 1000 else f"{ms} ms")


def dose_colors(doses):
    stim = sorted(d for d in doses if d > 0)
    idx = np.linspace(0, len(BLUES) - 1, len(stim)).round().astype(int) if len(stim) > 1 else [len(BLUES) - 1]
    c = {d: BLUES[i] for d, i in zip(stim, idx)}
    if 0 in doses:
        c[0] = GRAY
    return c


def style(ax, title=None, ylabel=None):
    ax.set_facecolor("white")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2); ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=INK2, labelsize=8, width=0.8)
    ax.grid(axis="y", color=GRID, linewidth=0.8); ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10, loc="left")
    if ylabel:
        ax.set_ylabel(ylabel, color=INK2, fontsize=9)


def direct_labels(ax, ends, min_gap_frac=0.06):
    """Label line ends at the right edge, nudged apart so labels don't collide."""
    if not ends:
        return
    lo, hi = ax.get_ylim(); gap = (hi - lo) * min_gap_frac
    ends = sorted(ends, key=lambda e: e[1])
    ys = [e[1] for e in ends]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    x = ax.get_xlim()[1]
    for (label, _, _), y in zip(ends, ys):
        ax.annotate(label, (x, y), xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8, color=INK2, annotation_clip=False)


def summarize(t, lay):
    t = t.copy()
    t["row"], t["col"] = t.well.str[0], t.well.str[1:].astype(int)
    t["line"], t["dose_ms"] = t.row.map(lay["line"]), t.col.map(lay["dose_ms"])
    t["tpi"] = t.tp.str[1:].astype(int); t["hours"] = t.tpi * lay["hours_per_tp"]
    t["ratio_median"] = t.rfp_median / t.gfp_median
    t["ratio_mean"] = t.rfp_mean / t.gfp_mean
    t["ratio_bgsub"] = (t.rfp_median - t.bg_rfp_median) / (t.gfp_median - t.bg_gfp_median)
    keys = ["well", "line", "dose_ms", "tp", "tpi", "hours"]
    agg = {}
    for m in list(METRICS) + ["fg_frac", "gfp_median", "rfp_median", "gfp_mean", "rfp_mean"]:
        agg[f"{m}"] = (m, "mean"); agg[f"{m}_sd"] = (m, "std")
        agg[f"{m}_q25"] = (m, lambda v: v.quantile(0.25)); agg[f"{m}_q75"] = (m, lambda v: v.quantile(0.75))
    agg["n_tiles"] = ("tile", "size")
    agg["n_fallback"] = ("s_source", lambda v: int((v != "tile").sum()))
    w = t.groupby(keys).agg(**agg).reset_index()
    w["ratio_pooled"] = (t.assign(r=t.rfp_mean * t.n_fg_px, g=t.gfp_mean * t.n_fg_px)
                         .groupby(keys)[["r", "g"]].sum().pipe(lambda d: d.r / d.g).values)
    for m in METRICS:
        w[f"{m}_sem"] = w[f"{m}_sd"] / np.sqrt(w.n_tiles)
        base = w[w.tpi == w.groupby("well").tpi.transform("min")].set_index("well")[m]
        w[f"{m}_fc"] = w[m] / w.well.map(base)
    return t.sort_values(["well", "tpi", "tile"]), w.sort_values(["well", "tpi"])


def condition_table(w):
    rows = []
    for m in METRICS:
        for (d, tpi), g in w.groupby(["dose_ms", "tpi"]):
            for col in (m, f"{m}_fc"):
                rows.append(dict(metric=col, dose_ms=d, tpi=tpi, hours=g.hours.iloc[0], n_lines=len(g),
                                 mean=g[col].mean(), sem=g[col].std() / np.sqrt(len(g)) if len(g) > 1 else np.nan))
    return pd.DataFrame(rows)


def fig_by_line(w, metric, fc, colors, path, title, source="Per-tile foreground readout"):
    lines = sorted(w.line.unique())
    col = f"{metric}_fc" if fc else metric
    fig, axes = plt.subplots(1, len(lines) + 1, figsize=(3.3 * (len(lines) + 1) + 0.8, 3.6), sharey=True)
    for ax, ln in zip(axes, lines):
        sub = w[w.line == ln]; ends = []
        for d in sorted(sub.dose_ms.unique()):
            for well, g in sub[sub.dose_ms == d].groupby("well"):
                g = g.sort_values("hours")
                if not fc:
                    ax.fill_between(g.hours, g[f"{metric}_q25"], g[f"{metric}_q75"], color=colors[d], alpha=0.18, lw=0)
                ax.plot(g.hours, g[col], color=colors[d], lw=2)
                ends.append((f"{dose_label(d)} ({well})" if sub.dose_ms.eq(d).sum() > g.shape[0] else dose_label(d),
                             g[col].iloc[-1], d))
        style(ax, ln, None); ax.set_xlabel("hours", color=INK2, fontsize=9)
        if fc:
            ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
        direct_labels(ax, ends)
    ax = axes[-1]; ends = []
    for d in sorted(w.dose_ms.unique()):
        g = w[w.dose_ms == d].groupby("hours")[col].agg(["mean", "std", "size"]).reset_index()
        sem = g["std"] / np.sqrt(g["size"])
        ax.fill_between(g.hours, g["mean"] - sem, g["mean"] + sem, color=colors[d], alpha=0.18, lw=0)
        ax.plot(g.hours, g["mean"], color=colors[d], lw=2)
        ends.append((dose_label(d), g["mean"].iloc[-1], d))
    style(ax, "All lines (mean ± SEM)"); ax.set_xlabel("hours", color=INK2, fontsize=9)
    if fc:
        ax.axhline(1, color=INK2, lw=0.8, ls=(0, (3, 3)))
    direct_labels(ax, ends)
    axes[0].set_ylabel(f"{METRICS[metric]}{', fold vs T0' if fc else ''}", color=INK2, fontsize=9)
    handles = [plt.Line2D([], [], color=colors[d], lw=2, label=dose_label(d)) for d in sorted(colors)]
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=8, title="blue light dose",
               title_fontsize=8)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    band = "" if fc else "  Band = IQR across the 9 tiles of the well."
    fig.text(0.01, 0.005, f"{source}; one well per line × dose.{band}", fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.97, 0.9)); fig.savefig(path, dpi=150); plt.close(fig)


def fig_wells(t, w, metric, lay, colors, path, title):
    rows, cols = sorted(t.row.unique()), sorted(t.col.unique())
    fig, axes = plt.subplots(len(rows), len(cols), figsize=(2.6 * len(cols), 2.1 * len(rows)), sharex=True, sharey=True)
    axes = np.atleast_2d(axes)
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            ax = axes[i, j]; well = f"{r}{c}"; d = lay["dose_ms"][c]
            for _, g in t[t.well == well].groupby("tile"):
                ax.plot(g.hours, g[metric], color="#c3c2b7", lw=0.8)
            g = w[w.well == well]
            ax.plot(g.hours, g[metric], color=colors[d], lw=2)
            style(ax, f"{well} · {lay['line'][r]} · {dose_label(d)}")
            ax.title.set_fontsize(8)
            if int(g.n_fallback.sum()):
                ax.text(0.02, 0.95, f"{int(g.n_fallback.sum())} fallback tiles", transform=ax.transAxes, fontsize=6.5,
                        color=INK2, va="top")
    for ax in axes[-1]:
        ax.set_xlabel("hours", color=INK2, fontsize=8)
    for ax in axes[:, 0]:
        ax.set_ylabel(METRICS[metric], color=INK2, fontsize=7.5)
    fig.suptitle(title + "  (gray = individual tiles, colored = well mean of 9 tiles)", x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96)); fig.savefig(path, dpi=140); plt.close(fig)


def fig_components(w, colors, path, title):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, m, lab in zip(axes, ["fg_frac", "gfp_median", "rfp_median"],
                          ["foreground fraction", "GFP median over foreground", "RFP median over foreground"]):
        ends = []
        for d in sorted(w.dose_ms.unique()):
            g = w[w.dose_ms == d].groupby("hours")[m].agg(["mean", "std", "size"]).reset_index()
            sem = g["std"] / np.sqrt(g["size"])
            ax.fill_between(g.hours, g["mean"] - sem, g["mean"] + sem, color=colors[d], alpha=0.18, lw=0)
            ax.plot(g.hours, g["mean"], color=colors[d], lw=2)
            ends.append((dose_label(d), g["mean"].iloc[-1], d))
        style(ax, lab); ax.set_xlabel("hours", color=INK2, fontsize=9)
        direct_labels(ax, ends)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK)
    fig.text(0.01, 0.005, "Mean ± SEM across cell lines, per dose. Raw counts (no background subtraction).",
             fontsize=7.5, color=INK2)
    fig.tight_layout(rect=(0, 0.03, 0.97, 0.93)); fig.savefig(path, dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tiles"); ap.add_argument("out"); ap.add_argument("--exp", default="hevo-pmsG-1")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lay = LAYOUT[a.exp]; tag = a.exp.replace("hevo-", "")
    t, w = summarize(pd.read_csv(a.tiles), lay)
    w.to_csv(f"{a.out}/well_tp.csv", index=False)
    condition_table(w).to_csv(f"{a.out}/condition_tp.csv", index=False)
    colors = dose_colors(set(w.dose_ms))
    fig_by_line(w, "ratio_median", False, colors, f"{a.out}/{tag}_ratio_median.png", f"{tag}: RFP/GFP over foreground, by line and dose")
    fig_by_line(w, "ratio_median", True, colors, f"{a.out}/{tag}_ratio_median_fc.png", f"{tag}: RFP/GFP fold change vs T0")
    fig_by_line(w, "ratio_mean", False, colors, f"{a.out}/{tag}_ratio_mean.png", f"{tag}: RFP/GFP (mean-based) over foreground")
    fig_by_line(w, "ratio_bgsub", False, colors, f"{a.out}/{tag}_ratio_bgsub.png", f"{tag}: background-subtracted RFP/GFP")
    fig_wells(t, w, "ratio_median", lay, colors, f"{a.out}/{tag}_wells_tiles.png", f"{tag}: RFP/GFP per well")
    fig_components(w, colors, f"{a.out}/{tag}_components.png", f"{tag}: what drives the ratio")
    print(f"{len(t)} tiles, {w.well.nunique()} wells, {w.tp.nunique()} timepoints; "
          f"{int(w.n_fallback.sum())} tile-timepoints used a fallback background")


if __name__ == "__main__":
    main()
