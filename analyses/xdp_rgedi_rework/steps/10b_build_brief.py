"""Step 10b: a five-minute brief for Julia, built from the same step outputs as the evidence document.

One page: the bottom line, the decisions that need her call, and three short sections (thresholds from S.A.,
XDP vs control death, GFP dropout), each with one or two purpose-built figures. Every number is read from a
step output; nothing is typed by hand. Writes <out>/docs/RGEDI_brief_<date>.html and an artifact copy without
the document wrapper.
Usage: python -I steps/10b_build_brief.py
"""

import base64
import html
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import norm  # noqa: E402

from lib import data, figs  # noqa: E402
from lib.sets import PRIMARY as P  # noqa: E402
from lib import report as R  # noqa: E402
from lib.paths import LOCAL_OUT, RAW_CACHE, RUN  # noqa: E402

E = html.escape
S = figs.LINE_SHORT
plt = figs.plt
OUT = LOCAL_OUT / "docs" / "brief"
EVIDENCE_URL = "https://claude.ai/artifact/3fGVTSxnJbLcFHFaiLgRZD"
XDP_PAIRS = [("XDP-33109.2B", "CTR-33114.C"), ("XDP-33363.D", "CTR-33114.C"),
             ("XDP-33109.2B", "CTR-33113.2I"), ("XDP-33363.D", "CTR-33113.2I")]


def png(fig, name: str) -> str:
    path = figs.save(fig, OUT / name, dpi=160)
    return base64.b64encode(Path(path).read_bytes()).decode()


def fig_block(b64: str, alt: str, cap: str) -> str:
    return (f'<figure><img class="plate" alt="{E(alt)}" src="data:image/png;base64,{b64}">'
            f'<figcaption>{cap}</figcaption></figure>')


def rng(a, b, nd=2, pct=False):
    f = (lambda v: f"{100 * v:.{max(nd - 2, 0)}f}%") if pct else (lambda v: f"{v:.{nd}f}")
    return f(a) if abs(a - b) < 10 ** -nd else f"{f(a)}–{f(b)}"


# ------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------
def fig_example(thr: pd.DataFrame, direct: pd.DataFrame):
    """One line in one experiment: untreated vs S.A.-killed cells, with the cutoff that S.A. puts in place."""
    exp, line = "XDP11", "CTR-33114.C"
    d = data.load(exp, columns=["line", "condition", "tp", "log10_ratio", "in_scope"])
    fit_tp = int(str(direct[(direct.exp == exp) & (direct.line == line)].fit_tps.iloc[0]).split(",")[0])
    d = d[(d.line == line) & (d.tp == fit_tp) & d.log10_ratio.notna()]
    t = thr[(thr.exp == exp) & (thr.line == line)].iloc[0]
    bins = np.linspace(-2.3, 0.3, 90)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    for cond, col, lab, fill in (("untreated", figs.UT, "untreated wells", False), ("sa", figs.SA, "S.A. wells (killed)", True)):
        x = d[d.condition == cond].log10_ratio.to_numpy()
        ax.hist(x, bins=bins, density=True, histtype="stepfilled" if fill else "step", color=col,
                alpha=0.35 if fill else 1, lw=1.4, label=f"{lab}, n = {len(x):,}")
    ax.axvspan(np.log10(t.W90_live), np.log10(t.W90_dead), color=figs.AMBIG, alpha=0.9, lw=0, zorder=0)
    ax.axvline(np.log10(t.D_live), color=figs.LIVE, lw=1.6, ls="--")
    top = ax.get_ylim()[1]
    box = dict(facecolor=figs.SURFACE, edgecolor="none", alpha=0.9, pad=1.5)
    ax.text(np.log10(t.W90_live) - 0.02, top * 0.93, f"live below\n{t.W90_live:.3f}", color=figs.LIVE, fontsize=8, va="top",
            ha="right", bbox=box)
    ax.text(np.log10(t.W90_dead) + 0.02, top * 0.93, f"dead above\n{t.W90_dead:.3f}", color=figs.DEAD, fontsize=8, va="top",
            bbox=box)
    ax.text(np.log10(t.D_live), top * 0.62, "buffer\nzone", color=figs.INK2, fontsize=7.5, ha="center")
    figs.log_ratio_axis(ax, "x", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xlabel("GEDI ratio (RFP / GFP), log scale")
    ax.set_ylabel("share of cells")
    ax.set_yticks([])
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title(f"{exp} {S[line]}, T{fit_tp} (the image after dosing)", loc="left", fontsize=9.5)
    return fig, exp, line, fit_tp, t


def fig_curves(direct: pd.DataFrame, exp: str = "XDP11", line: str = "CTR-33114.C"):
    """How the buffer zone is found: the two fitted curves, their crossing and the two 9-to-1 points."""
    r = direct[(direct.exp == exp) & (direct.line == line)].iloc[0]
    x = np.linspace(-2.0, 0.1, 600)
    live = norm.pdf(x, r.mode_log, r.sd_inner)
    dead = norm.pdf(x, r.dead_mu_log, r.dead_sd)
    c, lo, hi = np.log10(r.D_live), np.log10(r.W90_live), np.log10(r.W90_dead)
    fig, ax = plt.subplots(figsize=(7.2, 3.3))
    ax.axvspan(lo, hi, color=figs.AMBIG, alpha=0.9, lw=0, zorder=0)
    ax.plot(x, live, color=figs.LIVE, lw=2.2, label="live curve (untreated cells)")
    ax.plot(x, dead, color=figs.DEAD, lw=2.2, label="dead curve (S.A.-killed cells)")
    box = dict(facecolor=figs.SURFACE, edgecolor="none", alpha=0.9, pad=1.5)
    ymax = max(live.max(), dead.max())
    for xv, txt, col, ha in ((lo, f"live curve 9× higher\n{r.W90_live:.3f}", figs.LIVE, "right"),
                             (c, f"equal height\n{r.D_live:.3f}", figs.INK2, "center"),
                             (hi, f"dead curve 9× higher\n{r.W90_dead:.3f}", figs.DEAD, "left")):
        ax.axvline(xv, color=col, lw=1.2, ls="--")
        ax.text(xv + (-0.02 if ha == "right" else 0.02 if ha == "left" else 0), ymax * (1.2 if ha == "center" else 1.42),
                txt, color=col, fontsize=8, ha=ha, va="top", bbox=box)
    ax.set_ylim(0, ymax * 1.46)
    figs.log_ratio_axis(ax, "x", [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0])
    ax.set_xlabel("GEDI ratio (RFP / GFP), log scale")
    ax.set_ylabel("height of curve")
    ax.set_yticks([])
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.8), fontsize=8)
    ax.set_title(f"{exp} {S[line]}: the two fitted curves and the buffer zone (shaded)", loc="left", fontsize=9.5)
    return fig


def fig_t1(thr: pd.DataFrame):
    """T1 death per line x experiment: forced 9.1% by the September rule vs measured with the new cutoffs."""
    t = thr.copy()
    t["lab"] = t.exp + " " + t.line.map(S)
    t = t.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(t))
    ax.axvline(9.1, color=figs.DEAD, lw=2, ls="--")
    ax.text(9.25, len(t) - 0.4, "September rule: 9.1% for every line,\nby construction", color=figs.DEAD, fontsize=8, va="top")
    for i, r in t.iterrows():
        col = figs.LINE_COLOR[r.line]
        ax.plot([0, 100 * r[f"T1_death_{P}"]], [i, i], color=figs.GRID, lw=1, zorder=1)
        ax.scatter(100 * r[f"T1_death_{P}"], i, s=34, color=col, zorder=3,
                   edgecolor=col, facecolor=col if r.source.startswith("direct") else "white", lw=1.4)
    ax.set_yticks(y)
    ax.set_yticklabels(t.lab, fontsize=7.5)
    ax.set_xlabel("untreated cells called dead at T1 (%)")
    ax.set_xlim(0, max(14, 100 * t[f"T1_death_{P}"].max() + 1))
    ax.scatter([], [], s=34, color=figs.INK2, label="cutoff from this experiment's S.A. wells")
    ax.scatter([], [], s=34, facecolor="white", edgecolor=figs.INK2, label="cutoff transferred from other experiments")
    ax.legend(loc="lower right", fontsize=7.5)
    return fig


def fig_forest(per: pd.DataFrame, pooled: pd.DataFrame):
    """XDP vs each control: per-experiment odds ratios and the two pooled estimates."""
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharex=True)
    for ax, ref in zip(axes, ["CTR-33114.C", "CTR-33113.2I"]):
        yl, yt = [], []
        y = 0
        for cmp_ in ["XDP-33109.2B", "XDP-33363.D"]:
            col = figs.LINE_COLOR[cmp_]
            g = per[(per.Compare == cmp_) & (per.Reference == ref)].copy()
            g["exp"] = g.Experiment.str.replace("-RGEDI", "")
            g = g.sort_values("exp", key=lambda s: s.map({e: i for i, e in enumerate(figs.EXP_ORDER)}))
            p = pooled[(pooled.Compare == cmp_) & (pooled.Reference == ref)].iloc[0]
            ax.errorbar(p.mridpool_OR, y, xerr=[[p.mridpool_OR - p.mridpool_lo], [p.mridpool_hi - p.mridpool_OR]], fmt="D",
                        mfc="white", color=col, ms=6, capsize=3, lw=1.4)
            yl.append(y)
            yt.append(f"{S[cmp_]}: MRID as published")
            y -= 1
            ax.errorbar(p.lxe_OR, y, xerr=[[p.lxe_OR - p.lxe_lo], [p.lxe_hi - p.lxe_OR]], fmt="D", color=col, ms=7,
                        capsize=3, lw=2)
            yl.append(y)
            yt.append(f"{S[cmp_]}: allowing experiments to differ")
            y -= 1
            for r in g.itertuples():
                ax.errorbar(r.OR, y, xerr=[[r.OR - r.lower], [r.upper - r.OR]], fmt="o", color=col, ms=3.5, capsize=0,
                            lw=1, alpha=0.75)
                yl.append(y)
                yt.append(f"   {r.exp}")
                y -= 1
            y -= 0.6
        ax.axvline(1, color=figs.INK2, ls=":", lw=1)
        ax.set_xscale("log")
        figs.or_axis(ax, "x")
        ax.set_yticks(yl)
        ax.set_yticklabels(yt, fontsize=7.5)
        ax.set_title(f"vs control {S[ref]}", loc="left")
        ax.set_xlabel("odds ratio of death (> 1: XDP line dies more)")
    fig.tight_layout()
    return fig


def fig_capture(cap: pd.DataFrame):
    """Share of S.A.-killed cells still seen as dead, per line x experiment (cases passing the validity check)."""
    g = cap[(cap.set == "D_sd") & (cap.variant.isin(["primary", "t9"])) & cap.valid.astype(bool)]   # dropout: single cutoff
    fig, ax = plt.subplots(figsize=(7.2, 3.0))
    x, ticks, labs = 0, [], []
    for exp, var in (("XDP3", "primary"), ("XDP10", "t9"), ("XDP10", "primary"), ("XDP11", "primary")):
        e = g[(g.exp == exp) & (g.variant == var)]
        start = x
        for line in figs.LINE_ORDER:
            r = e[e.line == line]
            if not len(r):
                continue
            r = r.iloc[0]
            ax.errorbar(x, r.c, [[r.c - r.c_lo], [r.c_hi - r.c]], fmt="o", color=figs.LINE_COLOR[line], ms=6, capsize=2)
            ticks.append(x)
            labs.append(S[line])
            x += 1
        if x > start:
            ax.text((start + x - 1) / 2, 1.04, f"{exp} T{int(e.t.iloc[0])}",
                    transform=ax.get_xaxis_transform(), ha="center", fontsize=8.5, color=figs.INK2, fontweight="bold")
        x += 0.8
    ax.axhline(1, color=figs.INK2, ls=":", lw=1)
    ax.text(ax.get_xlim()[0] + 0.1, 1.02, "all killed cells still seen", fontsize=7.5, color=figs.INK2, va="bottom")
    ax.set_ylim(0, 1.15)
    ax.set_xticks(ticks)
    ax.set_xticklabels(labs, fontsize=8)
    ax.set_ylabel("share of S.A.-killed cells\nstill seen as dead")
    return fig


def fig_corrected(po: pd.DataFrame):
    """Pooled XDP-vs-control odds ratio with and without the dropout correction (primary set)."""
    fig, ax = plt.subplots(figsize=(7.2, 2.8))
    y = 0
    yl, yt = [], []
    for cmp_, ref in XDP_PAIRS:
        g = po[(po.set == P) & (po.Compare == cmp_) & (po.Reference == ref)]
        for kind, mk, col, off in (("MRID (uncorrected)", "o", figs.DEAD, 0.17),
                                   ("S.A.-corrected, D27 gated", "s", figs.SA, -0.17)):
            p = g[g.kind == kind].iloc[0]
            ax.errorbar(p.OR, y + off, xerr=[[p.OR - p.lo], [p.hi - p.OR]], fmt=mk, color=col, ms=6, capsize=3,
                        label=("uncorrected" if kind.startswith("MRID") else "corrected for dropout") if y == 0 else None)
        yl.append(y)
        yt.append(f"{S[cmp_]} vs {S[ref]}")
        y -= 1
    ax.axvline(1, color=figs.INK2, ls=":", lw=1)
    ax.set_xscale("log")
    figs.or_axis(ax, "x")
    ax.set_yticks(yl)
    ax.set_yticklabels(yt, fontsize=8.5)
    ax.set_xlabel("pooled odds ratio of death, 95% CI (> 1: XDP line dies more)")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, frameon=False)
    return fig


# ------------------------------------------------------------------------------------------------------------
CSS = """
/* Layout: one reading column; a decision panel up front; three evidence sections, each a claim, a figure and
   the numbers behind it. Same identity as the full evidence document, quieter. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --fg: #16181d; --muted: #5a6070; --rule: #dcdfe6;
  --accent: #4a3aa7; --accent-soft: #ecebf8; --plate: #fcfcfb; --plate-ring: rgba(11,11,11,0.10);
  --yes: #1d6f31; --yes-bg: #e3f2e6; --no: #7d5200; --no-bg: #fbefd6;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --font-ui: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111318; --surface: #181b22; --fg: #e7e9ef; --muted: #a3a9b8; --rule: #2c303b;
  --accent: #a99cf2; --accent-soft: #252040; --plate-ring: rgba(255,255,255,0.14);
  --yes: #7fd394; --yes-bg: #15301d; --no: #f2c56b; --no-bg: #33280f; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111318; --surface: #181b22; --fg: #e7e9ef; --muted: #a3a9b8; --rule: #2c303b;
  --accent: #a99cf2; --accent-soft: #252040; --plate-ring: rgba(255,255,255,0.14);
  --yes: #7fd394; --yes-bg: #15301d; --no: #f2c56b; --no-bg: #33280f; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 16.5px; line-height: 1.6;
  padding-inline: 16px; padding-block: 0 56px; }
main { max-width: 880px; margin: 0 auto; display: grid; gap: 40px; }
header.title { padding-block: 40px 4px; display: grid; gap: 10px; }
.eyebrow { font-family: var(--font-ui); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); }
h1 { font-family: var(--font-display); font-weight: 600; font-size: clamp(30px, 5vw, 42px); line-height: 1.08; margin: 0;
  text-wrap: balance; }
h2 { font-family: var(--font-display); font-weight: 600; font-size: 25px; line-height: 1.2; margin: 0; text-wrap: balance; }
h3 { font-family: var(--font-ui); font-size: 15px; font-weight: 600; margin: 0; }
p { margin: 0; max-width: 68ch; }
.lede { font-size: 18.5px; color: var(--muted); max-width: 62ch; }
.meta { font-family: var(--font-ui); font-size: 13px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 18px; }
a { color: var(--accent); }
a:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
section { display: grid; gap: 16px; min-width: 0; }
.bottom { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 20px 22px; display: grid; gap: 12px; }
.bottom ul, .facts { margin: 0; padding-left: 20px; display: grid; gap: 8px; max-width: 70ch; }
.decisions { display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--rule); border-radius: 10px; padding: 16px 18px; display: grid;
  gap: 10px; align-content: start; min-width: 0; }
.card .q { font-family: var(--font-ui); font-size: 15px; font-weight: 600; line-height: 1.35; }
.card .rec { font-family: var(--font-ui); font-size: 13px; background: var(--accent-soft); color: var(--fg); border-radius: 6px;
  padding: 8px 10px; }
.card .rec b { color: var(--accent); }
.card .if { font-family: var(--font-ui); font-size: 13px; display: grid; gap: 4px; color: var(--muted); }
.tag { display: inline-block; font-family: var(--font-mono); font-size: 11.5px; letter-spacing: 0.02em; padding: 1px 7px;
  border-radius: 999px; background: var(--accent-soft); color: var(--accent); }
.claim { font-size: 18px; font-weight: 600; max-width: 62ch; }
figure { margin: 0; display: grid; gap: 8px; min-width: 0; }
img.plate { width: 100%; height: auto; background: var(--plate); border-radius: 8px; box-shadow: 0 0 0 1px var(--plate-ring);
  padding: 10px; box-sizing: border-box; }
figcaption { font-family: var(--font-ui); font-size: 13px; color: var(--muted); max-width: 75ch; }
.two { display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }
.num { font-variant-numeric: tabular-nums; }
.limits { border-top: 1px solid var(--rule); padding-top: 18px; display: grid; gap: 10px; }
footer { font-family: var(--font-ui); font-size: 12px; color: var(--muted); border-top: 1px solid var(--rule); padding-top: 14px; }
"""
FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:'
         'wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')


def card(tag: str, q: str, rec: str, yes: str, no: str) -> str:
    return (f'<div class="card"><span class="tag">{E(tag)}</span><div class="q">{q}</div>'
            f'<div class="rec"><b>Recommended:</b> {rec}</div>'
            f'<div class="if"><span><b>Yes</b> → {yes}</span><span><b>No</b> → {no}</span></div></div>')


def build(simple: bool = False) -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    thr = pd.read_csv(LOCAL_OUT / "10a_final_tables" / "thresholds" / "thresholds_all.csv")
    direct = pd.read_csv(LOCAL_OUT / "06_fit_thresholds" / "thresholds_direct.csv")
    s06 = R.step_json("06_fit_thresholds", "summary")
    s10 = R.step_json("10a_final_tables", "summary")
    audit = pd.read_csv(RAW_CACHE / "MRID-rework/audit/threshold_audit_ALL.csv")
    pooled = R.step_csv("13_or_figures", "pooled_or_by_set.csv")
    pooled = pooled[pooled.set == P]
    per = R.step_csv(f"12_logodds/{P}", "OR_time_averaged_per_experiment.csv").rename(columns=lambda c: c.split(".")[0])
    cap = R.step_csv("15_sa_capture", "capture.csv")
    po17 = R.step_csv("17_count_synthesis", "or_pooled_uncorrected_vs_corrected.csv")
    s17 = R.step_json("17_count_synthesis", "summary")
    c1 = R.step_csv("16_retention", "c1_pairs.csv")
    c2p = R.step_csv("16b_retention_fe", "c2_pooled.csv")

    # ---- numbers ---------------------------------------------------------------------------------------------
    sept_n = int((audit.baseline_death_pct.round(1) == 9.1).sum())
    t1 = thr[f"T1_death_{P}"]
    s6b = R.step_json("06b_window_cutoffs", "summary")
    u = direct[direct.usable]
    called_D = 1 - norm.cdf((np.log10(u.D_live) - u.dead_mu_log) / u.dead_sd)
    called_W = 1 - norm.cdf((np.log10(u.W90_dead) - u.dead_mu_log) / u.dead_sd)
    called_B = 1 - norm.cdf((np.log10(u.B_dead) - u.dead_mu_log) / u.dead_sd)
    xdp = pooled[pooled.Compare.str.startswith("XDP")]
    v13 = xdp[xdp.Reference == "CTR-33113.2I"]
    n_incl = int(((xdp.lxe_lo <= 1) & (xdp.lxe_hi >= 1)).sum())
    i2 = xdp.meta_all_I2
    per_x = per[per.Compare.str.startswith("XDP") & per.Reference.isin(["CTR-33114.C", "CTR-33113.2I"])]
    above, below = int((per_x.lower > 1).sum()), int((per_x.upper < 1).sum())
    gate = cap[(cap.set == "D_sd") & (cap.variant == "primary") & cap.valid.astype(bool)]   # dropout: single cutoff
    t9 = cap[(cap.set == "D_sd") & (cap.variant == "t9") & (cap.exp == "XDP10")]
    t10 = gate[gate.exp == "XDP10"]
    agr = s17["agreement"][P]
    pc = po17[(po17.set == P) & po17.Compare.str.startswith("XDP")]
    ex = lambda kind: int(((pc.kind == kind) & ((pc.lo > 1) | (pc.hi < 1))).sum())  # noqa: E731
    r63 = c1[(c1.fit == P) & (c1.Compare == "XDP-33363.D") & (c1.Reference == "CTR-33114.C")].iloc[0]
    q63 = c2p[(c2p.fit == P) & (c2p.Compare == "XDP-33363.D") & (c2p.Reference == "CTR-33114.C")].iloc[0]

    # ---- figures ---------------------------------------------------------------------------------------------
    f_ex, ex_exp, ex_line, ex_tp, ex_t = fig_example(thr, direct)
    figA = png(f_ex, "brief_example.png")
    figB = png(fig_t1(thr), "brief_t1.png")
    figC = png(fig_forest(per_x, xdp), "brief_forest.png")
    figD = png(fig_capture(cap), "brief_capture.png")
    figE = png(fig_corrected(po17), "brief_corrected.png")
    figW = png(fig_curves(direct), "brief_curves.png")

    lx = lambda c, r: xdp[(xdp.Compare == c) & (xdp.Reference == r)].iloc[0]  # noqa: E731
    v14 = xdp[xdp.Reference == "CTR-33114.C"]
    n_m = int((v14.mridpool_lo > 1).sum())
    mrid_txt = (f"finds more death in {'both XDP lines' if n_m == 2 else S[v14[v14.mridpool_lo > 1].Compare.iloc[0]]} than in 14C"
                if n_m else "finds no clear XDP-vs-14C difference either")
    death = R.step_csv("07_validity", "reference_death_by_tp.csv")
    drop90 = float(death[(death.tp == 1) & (death.rule == P)].ambiguous.median())
    if simple:
        return build_simple(locals())
    out = ["<title>XDP Death Analysis Brief</title>", FONTS, f"<style>{CSS}</style>", "<main>"]
    out.append(f"""<header class="title"><span class="eyebrow">Finkbeiner lab · XDP iPSC neurons · RGEDI survival · for Julia Kaye</span>
<h1>XDP death analysis: what the new methods show</h1>
<p class="lede">A five-minute summary of the threshold rework, the odds-ratio comparison and the GFP-dropout check, and the
four calls we need from you. Every number comes from the <a href="{EVIDENCE_URL}">full evidence document</a>, which
has the step-by-step detail.</p>
<div class="meta"><span>Austin Holub, {time.strftime('%d %b %Y')}</span><span>6 experiments · 4 lines (14C, 13I controls; 09B, 63D XDP)</span></div>
</header>""")
    out.append(f"""<section class="bottom" aria-label="Bottom line"><h2>Bottom line</h2><ul>
<li><b>The new cutoffs come from known-dead cells.</b> Cells killed with sodium arsenite (S.A.) show where dead cells sit
on the GEDI ratio, line by line. Between live and dead there is now a buffer zone: cells that are less than 90% likely to be
either are left out. The September rule set T1 death to 9.1% in every line by construction; the new cutoffs measure it
({rng(t1.min(), t1.max(), 3, True)} at T1).</li>
<li><b>XDP lines do not die reproducibly more than controls.</b> MRID's published model, which assumes every experiment
shows the same line difference, {mrid_txt}. But the experiments differ: per experiment the XDP-vs-control odds ratio
lies clearly above 1 in {above} and clearly below 1 in {below} of {len(per_x)} comparisons. Once that variation is allowed, all
{n_incl if n_incl == len(xdp) else f'{n_incl} of {len(xdp)}'} XDP-vs-control intervals include 1.</li>
<li><b>Dying cells do drop out of the images, a lot</b> - one or two images after S.A., only
{rng(gate.c.min(), gate.c.max(), 2, True)} of killed cells are still seen as dead. Correcting for that does not change
the conclusion above; whether dropout differs between lines cannot be settled with these images.</li>
</ul></section>""")
    out.append(f"""<section aria-label="Decisions"><h2>Your call</h2>
<p>Four decisions. Each card gives our recommendation and what happens either way.</p>
<div class="decisions">
{card("method", "Set live/dead cutoffs per line and experiment from S.A.-killed cells (transferred where S.A. gave no clear dead population)?",
      "yes. It replaces a rule that fixed death at 9.1% and ties every cutoff to cells we know are dead.",
      "MRID reruns use these cutoffs (done; Steps 11-13).", "we go back to one global pair or the September rule, with the 9.1% artefact.")}
{card("D28", "Cutoff rule: call a cell live or dead only when it is at least 90% likely to be one or the other, with the zone in between centred where live and dead are equally likely?",
      f"yes (Austin's choice). It drops a median {100 * drop90:.1f}% of untreated cells as undecided and still calls a median {100 * np.median(called_W):.0f}% of S.A.-killed cells dead (single cutoff {100 * np.median(called_D):.0f}%; the old 2.5-SD-both-sides rule B only {100 * np.median(called_B):.0f}%).",
      "the 90% window is the primary result; the 95% window and the single cutoff are shown as checks.", "back to one cutoff with every cell called, or to rule B.")}
{card("D25", "Should the pooled comparison allow the line difference to vary between experiments?",
      f"yes. The per-experiment differences vary far more than chance (I² {rng(i2.min(), i2.max(), 2, True)}); without this the intervals are much too narrow.",
      "the headline odds ratios are the wider ones (filled diamonds below).", "MRID's original model, which reports XDP > 14C with narrow intervals.")}
{card("D27", "Use the S.A.-based dropout correction only as a sensitivity check, not as the main result?",
      "yes. The correction changes with which image after dosing is used, so it is too unstable to replace the measured odds ratios.",
      "MRID odds ratios stay primary; the corrected values are shown beside them.", "the corrected odds ratios become primary (the rule we fixed in advance said so; it does not change the conclusion).")}
</div></section>""")

    # ---- section 1 -----------------------------------------------------------------------------------------
    out.append(f"""<section aria-labelledby="s1"><h2 id="s1">1 · Cutoffs from cells we know are dead</h2>
<p class="claim">S.A. adds a second, high-ratio population of dead cells; the buffer zone sits where it takes over from the live cells.</p>
{fig_block(figA, "Histogram of GEDI ratios, untreated vs S.A. wells", f"One line in one experiment ({ex_exp} {S[ex_line]}, T{ex_tp}). Grey outline: untreated wells; violet: S.A. wells after dosing. Dashed line: the centre ({ex_t.D_live:.3f}), where a cell is equally likely to be live or dead. Shaded: the buffer zone ({ex_t.W90_live:.3f}–{ex_t.W90_dead:.3f}), where a cell is less than 90% likely to be either; those cells are left out.")}
<ul class="facts">
<li>{s06['usable']} of {s06['cases']} line × experiment cases with S.A. wells show a clear S.A.-dead population; their centres are
{rng(s06['D_range'][0], s06['D_range'][1], 2)} (bootstrap intervals about ±0.01) and their buffer zones {rng(*s6b['W90']['fold_width_range'])}-fold wide:
narrow where live and dead separate well, wide where they overlap. The other {s10['transfer']} cases (including all of XDP8, which has
no S.A. images) take cutoffs transferred from the same line's other experiments.</li>
<li>Cutoffs are set per line and per experiment, so a brighter or dimmer line is not mistaken for more or less death.</li>
</ul>
{fig_block(figB, "T1 death per line and experiment", f"Share of untreated cells called dead at T1 with the new cutoffs and buffer zone (dots) against the September rule, which gave 9.1% in {sept_n} of {len(audit)} audited line × experiment rows because it was defined from each line's own T1 cells.")}
</section>""")

    # ---- section 2 -----------------------------------------------------------------------------------------
    o09, o63 = lx("XDP-33109.2B", "CTR-33114.C"), lx("XDP-33363.D", "CTR-33114.C")
    srcmap = {(e, l): s_ for e, l, s_ in zip(thr.exp, thr.line, thr.source)}
    top = per_x.assign(exp=per_x.Experiment.str.replace("-RGEDI", "")).sort_values("OR", ascending=False).head(3)
    assert all("transfer" in srcmap[(r.exp, r.Compare)] or "transfer" in srcmap[(r.exp, r.Reference)] for r in top.itertuples()), \
        "largest odds ratios no longer all involve a transferred cutoff - rewrite this bullet"
    out.append(f"""<section aria-labelledby="s2"><h2 id="s2">2 · Do the XDP lines die more than the controls?</h2>
<p class="claim">Not reproducibly. The answer flips between experiments and depends on which control line is used.</p>
{fig_block(figC, "Odds ratios of death for each XDP line against each control", "Small dots: one experiment each (95% CI). Open diamond: MRID's model as published, which treats every experiment as showing the same difference. Filled diamond: the same model allowing the difference to vary between experiments (D25). Cutoffs with the 90% buffer zone; odds ratios above 1 mean the XDP line has more dead cells.")}
<ul class="facts">
<li>Against 14C, MRID as published gives 09B {o09.mridpool_OR:.2f} [{o09.mridpool_lo:.2f}, {o09.mridpool_hi:.2f}] and 63D
{o63.mridpool_OR:.2f} [{o63.mridpool_lo:.2f}, {o63.mridpool_hi:.2f}]. Allowing experiments to differ: 09B
{o09.lxe_OR:.2f} [{o09.lxe_lo:.2f}, {o09.lxe_hi:.2f}], 63D {o63.lxe_OR:.2f} [{o63.lxe_lo:.2f}, {o63.lxe_hi:.2f}].</li>
<li>Against 13I the XDP lines do not show more death (pooled {rng(v13.lxe_OR.min(), v13.lxe_OR.max())}); 13I was imaged only
in XDP5, XDP10 and XDP11.</li>
<li>The {len(top)} largest single-experiment odds ratios ({', '.join(f"{S[r.Compare]} vs {S[r.Reference]} {r.exp} {r.OR:.1f}" for r in top.itertuples())})
all involve at least one transferred cutoff rather than one fitted on that experiment's own S.A. wells.</li>
</ul></section>""")

    # ---- section 3 -----------------------------------------------------------------------------------------
    t9s = "; ".join(f"{S[r.line]} {r.c:.2f}" + ("" if r.valid else " (fails the validity check, not plotted)") for r in t9.sort_values("line", key=lambda s: s.map({l: i for i, l in enumerate(figs.LINE_ORDER)})).itertuples())
    t10s = rng(t10.c.min(), t10.c.max())
    out.append(f"""<section aria-labelledby="s3"><h2 id="s3">3 · Dying cells that drop out of the images</h2>
<p class="claim">{rng(1 - gate.c.max(), 1 - gate.c.min(), 2, True)} of S.A.-killed cells are gone from the images one or two
images after dosing; correcting for this does not change the answer in section 2.</p>
<p>The software finds cells by their GFP. A dying cell loses GFP, so MRID never sees it. S.A. wells let us measure this:
compared with untreated wells of the same line on the same plate, we count how many live cells S.A. removed and how many of
those are still seen as dead.</p>
<div class="two">
{fig_block(figD, "Share of S.A.-killed cells still seen as dead", f"Measured with the single cutoff, so that only dropout counts (with the buffer zone, killed cells inside the zone would also count as not seen). Cases that pass the validity check. XDP10 read at T9 (one image after dosing: {t9s}) and at T10 (two images: {t10s}) gives different line orders, so this share is not a stable property of a line.")}
{fig_block(figE, "Pooled odds ratios with and without the dropout correction", "Each XDP line against each control, pooled over experiments with small-sample intervals: uncorrected (orange) and corrected with each line's S.A. capture share (violet).")}
</div>
<ul class="facts">
<li>With or without the correction, {'every XDP-vs-control interval includes 1' if ex('MRID (uncorrected)') + ex('S.A.-corrected, D27 gated') == 0 else 'see the full document for which intervals exclude 1'}.</li>
<li>Live-cell counts are a second, independent readout. They agree with the GEDI odds ratios in direction in only
{agr['sign_agree']} of {agr['n']} line pair × experiment comparisons, about what chance gives, so neither readout can be read as
the death difference on its own. Counts alone suggest 63D loses live cells somewhat faster than 14C
({1 / r63.ratio_per_day ** 7:.2f}× per week in the pooled model), but not once experiments are pooled with small-sample
intervals ({1 / q63.ratio_per_day ** 7:.2f} [{1 / q63.upper ** 7:.2f}, {1 / q63.lower ** 7:.2f}]).</li>
<li>What would make a dropout correction possible: imaging S.A. wells at several short intervals after dosing, or segmenting
on RFP or both channels so dying cells stay detectable.</li>
</ul></section>""")

    out.append("""<section class="limits" aria-labelledby="lim"><h3 id="lim">What this does not show</h3>
<ul class="facts">
<li>S.A. death is acute; disease-related death is slow. The cutoffs only need S.A. to show where the dead state of the sensor
sits; whether spontaneously dying cells drop out at the same rate is not known.</li>
<li>XDP10 has only 2 S.A. wells per line, all at the plate edge; XDP3's dose time is inferred; XDP1 has no untreated wells.</li>
<li>The rules for cutoffs and the dropout check were fixed before the fits. Two changes were made after seeing results and are
logged as decisions (D18-D20 for the cutoff rules, D27 for the dropout correction).</li>
</ul></section>""")
    out.append(f'<footer>Built {time.strftime("%Y-%m-%d %H:%M")} from the step outputs of the XDP RGEDI rework '
               f'(analyses/xdp_rgedi_rework, analysis run {RUN}). Full detail, checks and decision log: '
               f'<a href="{EVIDENCE_URL}">XDP RGEDI Threshold Evidence</a>.</footer></main>')
    return "\n".join(out)


def build_simple(v: dict) -> str:
    """The same brief in Simple English (ASD-STE100 style): short sentences, one fact each, no hedging."""
    pct = lambda a, b: rng(a, b, 2, True)  # noqa: E731
    xdp = v["xdp"]
    lx = lambda c, r: xdp[(xdp.Compare == c) & (xdp.Reference == r)].iloc[0]  # noqa: E731
    o09, o63 = lx("XDP-33109.2B", "CTR-33114.C"), lx("XDP-33363.D", "CTR-33114.C")
    v14 = xdp[xdp.Reference == "CTR-33114.C"]
    n_m = int((v14.mridpool_lo > 1).sum())
    s6b, s06, s10 = v["s6b"], v["s06"], v["s10"]
    gate, t9, t10, agr, ex = v["gate"], v["t9"], v["t10"], v["agr"], v["ex"]
    t1, n_incl, i2 = v["t1"], v["n_incl"], v["i2"]
    t9s = ", ".join(f"{S[r.line]} {100 * r.c:.0f}%" for r in t9[t9.valid.astype(bool)].itertuples())
    ex_t = v["ex_t"]
    mrid = (f"finds more death in {'both XDP lines' if n_m == 2 else 'one XDP line'} than in control 14C" if n_m
            else "finds no clear difference from control 14C")
    allin = ex("MRID (uncorrected)") + ex("S.A.-corrected, D27 gated") == 0
    wide = ("p, .lede, .claim, .facts, .bottom ul, figcaption { max-width: none; }\n"
            "header.title { padding-block: 32px 0; }")
    out = ["<title>XDP Death Brief in Simple English</title>", FONTS, f"<style>{CSS}\n{wide}</style>", "<main>"]
    out.append(f"""<header class="title"><h1>XDP death analysis in simple English</h1>
<p class="lede">Six experiments and four lines: 14C and 13I are controls, 09B and 63D are XDP. The
<a href="{EVIDENCE_URL}">full evidence document</a> has every step and every check.</p>
</header>""")
    out.append(f"""<section class="bottom" aria-label="Bottom line"><h2>Bottom line</h2><ul>
<li><b>The new cutoffs come from cells that we know are dead.</b> Sodium arsenite (S.A.) kills cells. These dead cells
show where dead cells sit on the GEDI ratio. Cells close to the cutoff go into a buffer zone, and the analysis leaves them
out. The old September rule set T1 death to 9.1% in every line. The new cutoffs measure T1 death: {pct(t1.min(), t1.max())}.</li>
<li><b>The XDP lines do not die more than the controls in a reproducible way.</b> MRID's published model {mrid}. That
model assumes that every experiment shows the same difference between two lines. The experiments do not agree. When the
model lets each experiment have its own difference, all {n_incl} XDP-vs-control intervals include 1.</li>
<li><b>Many dying cells disappear from the images.</b> One or two images after S.A., GEDI still sees only
{pct(gate.c.min(), gate.c.max())} of the killed cells as dead. A correction for this does not change the result above.
These images cannot show if the disappearance is different between lines.</li>
</ul></section>""")
    out.append(f"""<section aria-label="Decisions"><h2>Your decisions</h2>
<div class="decisions">
{card("method", "Do we set the live and dead cutoffs from S.A.-killed cells, for each line in each experiment?",
      "yes. Then every cutoff comes from cells that we know are dead. The old rule fixed T1 death at 9.1% in every line.",
      "MRID uses these cutoffs. This work is done.", "we go back to one cutoff pair for all lines, or to the September rule.")}
{card("D28", "Do we leave out the cells that are less than 90% likely to be live, or less than 90% likely to be dead?",
      f"yes (Austin's choice). This leaves out {100 * v['drop90']:.1f}% of untreated cells (median). It still calls {100 * np.median(v['called_W']):.0f}% of S.A.-killed cells dead.",
      "the 90% zone is the main result. The 95% zone and the single cutoff are checks.", "we use one cutoff and call every cell live or dead.")}
{card("D25", "Do we let the difference between two lines change from experiment to experiment?",
      f"yes. The experiments disagree much more than chance can explain (I² {pct(i2.min(), i2.max())}). Without this step, the intervals are too narrow.",
      "the main odds ratios are the wide ones (the filled diamonds below).", "we report MRID's original model, which has narrow intervals.")}
{card("D27", "Do we use the dropout correction only as a check?",
      "yes. The correction changes when we measure at a different image after dosing. So it is not stable enough to be the main result.",
      "the MRID odds ratios stay the main result.", "the corrected odds ratios become the main result. The conclusion does not change.")}
</div></section>""")
    out.append(f"""<section aria-labelledby="s1"><h2 id="s1">1 · Cutoffs from cells that we know are dead</h2>
<p class="claim">S.A. adds a second group of cells with a high GEDI ratio. These cells are dead. The buffer zone sits between
this group and the live cells.</p>
<h3>How we find the buffer zone</h3>
<p>We do these steps for each line in each experiment.</p>
<ol class="facts">
<li>We take the untreated cells. We find the GEDI ratio where most of them sit (the live peak). We also measure how wide
this live group is. We measure the width only on the low side of the peak, because dead cells never reach that side.</li>
<li>We take the S.A. wells after dosing. A mixture model finds the group of S.A.-killed cells with a high ratio. We measure
the centre and the width of this dead group.</li>
<li>We draw each group as a bell curve with that centre and width (figure below). Each curve has the same total area.</li>
<li>At each GEDI ratio, we compare the heights of the two curves. If the dead curve is 9 times higher, a cell at that ratio
is 9 times more likely to be dead than live. That is 90% certainty.</li>
<li>The centre of the buffer zone is the ratio where the two curves have the same height.</li>
<li>The live edge is the ratio below the centre where the live curve is 9 times higher. The dead edge is the ratio above
the centre where the dead curve is 9 times higher.</li>
<li>Cells below the live edge are live. Cells above the dead edge are dead. The analysis leaves out the cells between the
two edges.</li>
<li>Some cases have no clear S.A.-dead group. For these cases, we measure how far the edges sit from the live peak in the same
line in other experiments. We then use that distance, in units of the live width.</li>
</ol>
{fig_block(v['figW'], "The two fitted curves and the buffer zone", f"{v['ex_exp']} {S[v['ex_line']]}. Blue: the curve for live cells, from untreated wells. Orange: the curve for dead cells, from S.A. wells. The dashed lines mark the live edge, the centre and the dead edge. The shaded zone is the buffer zone.")}
{fig_block(v['figA'], "Histogram of GEDI ratios, untreated wells and S.A. wells", f"One line in one experiment ({v['ex_exp']} {S[v['ex_line']]}, T{v['ex_tp']}). The grey line shows untreated wells. The violet bars show S.A. wells after dosing. Cells below {ex_t.W90_live:.3f} are live. Cells above {ex_t.W90_dead:.3f} are dead. The analysis leaves out the cells in the shaded buffer zone.")}
<ul class="facts">
<li>{s06['usable']} of {s06['cases']} cases with S.A. wells show a clear group of S.A.-dead cells. (A case is one line in one experiment.)</li>
<li>In these cases, the centre of the buffer zone is between {s06['D_range'][0]:.2f} and {s06['D_range'][1]:.2f}.</li>
<li>The zone is {rng(*s6b['W90']['fold_width_range'])} times wide. It is narrow when live and dead cells separate well.</li>
<li>The other {s10['transfer']} cases use cutoffs from the same line in other experiments. XDP8 had no S.A. images, so all its
cutoffs come from other experiments.</li>
</ul>
{fig_block(v['figB'], "T1 death for each line and experiment", f"Each dot shows the share of untreated cells that the new cutoffs call dead at T1. The dashed line is the September rule. That rule gave 9.1% in {v['sept_n']} of {len(v['audit'])} audited rows.")}
</section>""")
    out.append(f"""<section aria-labelledby="s2"><h2 id="s2">2 · Do the XDP lines die more than the controls?</h2>
<p class="claim">No, not in a reproducible way. The answer changes from experiment to experiment. It also changes with the
control line.</p>
{fig_block(v['figC'], "Odds ratios of death for each XDP line against each control", "Each small dot is one experiment. The open diamond is MRID's published model. The filled diamond lets each experiment have its own difference. An odds ratio above 1 means that the XDP line has more dead cells.")}
<ul class="facts">
<li>Against 14C, MRID's published model gives 09B {o09.mridpool_OR:.2f} [{o09.mridpool_lo:.2f}, {o09.mridpool_hi:.2f}] and 63D
{o63.mridpool_OR:.2f} [{o63.mridpool_lo:.2f}, {o63.mridpool_hi:.2f}].</li>
<li>When each experiment has its own difference, the values are 09B {o09.lxe_OR:.2f} [{o09.lxe_lo:.2f}, {o09.lxe_hi:.2f}] and 63D
{o63.lxe_OR:.2f} [{o63.lxe_lo:.2f}, {o63.lxe_hi:.2f}]. Both intervals include 1.</li>
<li>Against 13I, the XDP lines do not show more death. Only XDP5, XDP10 and XDP11 include 13I.</li>
</ul></section>""")
    out.append(f"""<section aria-labelledby="s3"><h2 id="s3">3 · Dying cells that disappear</h2>
<p class="claim">Many S.A.-killed cells disappear from the images. A correction for this does not change the answer in section 2.</p>
<p>The software finds cells by their GFP signal. A dying cell loses GFP. Then the software does not find it, and MRID does
not count it. S.A. wells let us measure this. We compare them with untreated wells of the same line on the same plate.</p>
<div class="two">
{fig_block(v['figD'], "Share of S.A.-killed cells that GEDI still sees as dead", f"Each dot is one line in one experiment. XDP10 at T9, one image after dosing, gives {t9s}. At T10, two images after dosing, all lines give {rng(100 * t10.c.min(), 100 * t10.c.max(), 0)}%. So this share is not a stable property of a line.")}
{fig_block(v['figE'], "Odds ratios with and without the dropout correction", "Orange: the odds ratio without correction. Violet: the odds ratio with the correction. Each XDP line is compared with each control, over all experiments.")}
</div>
<ul class="facts">
<li>{'With or without the correction, every XDP-vs-control interval includes 1.' if allin else 'The full document shows which intervals exclude 1.'}</li>
<li>Cell counts give a second measure of death. Counts and odds ratios agree in direction in only {agr['sign_agree']} of
{agr['n']} comparisons. This is what chance gives. So neither measure is reliable alone.</li>
</ul></section>""")
    out.append("""<section class="limits" aria-labelledby="lim"><h3 id="lim">What this does not show</h3>
<ul class="facts">
<li>S.A. kills cells fast. Disease kills cells slowly. We do not know if slowly dying cells disappear at the same rate.</li>
<li>XDP10 has only 2 S.A. wells per line, at the edge of the plate. XDP3 has no recorded dose time. XDP1 has no untreated wells.</li>
<li>We fixed the rules before the fits. We changed some rules after we saw results. We recorded each change as a decision
(D18-D20, D27 and D28).</li>
</ul></section>""")
    out.append("</main>")
    return "\n".join(out)


def main() -> None:
    simple = "--simple" in sys.argv
    body = build(simple)
    name = f"RGEDI_brief{'_simple' if simple else ''}_{time.strftime('%Y%m%d')}.html"
    docs = LOCAL_OUT / "docs"
    (docs / "artifact").mkdir(parents=True, exist_ok=True)
    (docs / "artifact" / name).write_text(body)
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>' + body + "</body></html>")
    (docs / name).write_text(page)
    print(f"wrote {docs / name} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
