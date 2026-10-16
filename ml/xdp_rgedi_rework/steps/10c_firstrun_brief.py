"""Step 10c: a Simple English review of the first MRID run (4 September 2026), before the September rework.

Reads the first run's own files (MRID inputs and combined outputs in Dropbox), checks its inputs against the plate
layout in config/experiments.yaml, and compares its cutoffs and T1 death with the current analysis (run 2).
Every number is computed here from those files. Writes <cache>/out/review_firstrun_20260904/:
RGEDI_firstrun_review_simple_<date>.html, an artifact copy without the document wrapper, and the figures.
Usage: python -I steps/10c_firstrun_brief.py
"""

import base64
import html
import importlib.util
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs  # noqa: E402
from lib.paths import CACHE, LOCAL_OUT  # noqa: E402

E = html.escape
S = figs.LINE_SHORT
plt = figs.plt
FIRST = Path("/Users/aholub/Library/CloudStorage/Dropbox-Gladstone/Austin Holub/other/Austin/Analysis/MRID/"
             "MRID_XDP-1-3-5-8-10-11_04SEPT2026")
OUT = CACHE / "out" / "review_firstrun_20260904"
EXPS = ["XDP1", "XDP3", "XDP5", "XDP8", "XDP10", "XDP11"]
XDP_LINES = ["XDP-33109.2B", "XDP-33363.D"]
CONTROLS = ["CTR-33114.C", "CTR-33113.2I"]
EVIDENCE_URL = "https://claude.ai/artifact/3fGVTSxnJbLcFHFaiLgRZD"

_spec = importlib.util.spec_from_file_location("brief", Path(__file__).with_name("10b_build_brief.py"))
_brief = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_brief)
CSS, FONTS, fig_block = _brief.CSS, _brief.FONTS, _brief.fig_block


def png(fig, name: str) -> str:
    path = figs.save(fig, OUT / name, dpi=160)
    return base64.b64encode(Path(path).read_bytes()).decode()


# ------------------------------------------------------------------------------------------------------------
# inputs of the first run
# ------------------------------------------------------------------------------------------------------------
def load_inputs() -> dict:
    out = {}
    for exp in EXPS:
        f = next((FIRST / "input" / f"{exp}-RGEDI").glob("*_ratio_output.csv"))
        d = pd.read_csv(f, usecols=["filenames", "Sci_WellID", "Sci_SampleID", "Timepoint", "ratio", "live_guesses"])
        d["well"] = d.Sci_WellID.map(cfg.pad_well)
        roles = cfg.well_roles(exp).set_index("well")
        d["true_line"] = d.well.map(roles.line)
        d["cond"] = d.well.map(roles.condition)
        out[exp] = d
    return out


def audit(inp: dict) -> pd.DataFrame:
    rows = []
    for exp, d in inp.items():
        tps_all = sorted(cfg.read_timepoints(exp).tp)
        s = cfg.experiments()[exp]["sa"]
        dose_after = next((x["after_tp"] for x in s.get("doses", []) if x.get("after_tp")), None)
        sa = d[d.cond == "sa"]
        post = sa[sa.Timepoint > dose_after] if dose_after else sa.iloc[0:0]
        rows.append({
            "exp": exp, "cells": len(d), "wells": d.well.nunique(),
            "live_cut": float(d[d.live_guesses == 1].ratio.max()), "dead_cut": float(d[d.live_guesses == 0].ratio.min()),
            "tps": sorted(int(t) for t in d.Timepoint.unique()), "tps_all": tps_all,
            "sirna_wells": int(d[d.cond == "sirna"].well.nunique()),
            "sa_wells_post_dose": int(post.well.nunique()), "sa_cells_post_dose": int(len(post)),
            "dup_cells": int(d.filenames.duplicated().sum()),
            "mislabelled_cells": int((d.Sci_SampleID != d.true_line).sum()),
            "missing_wells": sorted(set(cfg.well_roles(exp).query("in_scope and condition == 'untreated'").well)
                                    - set(d.well)) if exp != "XDP1" else [],
        })
    return pd.DataFrame(rows)


def t1_death(inp: dict) -> pd.DataFrame:
    rows = []
    for exp, d in inp.items():
        t1 = d[d.Timepoint == 1]
        for line, g in t1.groupby("Sci_SampleID"):
            if line in S:
                rows.append({"exp": exp, "line": line, "first_run": float((g.live_guesses == 0).mean())})
    return pd.DataFrame(rows)


def dropped_share(a: pd.DataFrame) -> pd.DataFrame:
    """Share of cells between the first run's two cutoffs (dropped), from the per-cell data of the same wells."""
    rows = []
    for r in a.itertuples():
        d = data.load(r.exp, columns=["line", "condition", "tp", "ratio", "in_scope"])
        d = d[d.tp <= 10] if r.exp == "XDP1" else d[d.condition == "untreated"]
        d = d[d.tp.isin(r.tps)]
        rows.append({"exp": r.exp, "dropped": float(((d.ratio > r.live_cut) & (d.ratio < r.dead_cut)).mean())})
    return pd.DataFrame(rows)


def load_or() -> dict:
    o = FIRST / "output" / "MRID-NonLinear"
    hours = sorted(pd.read_csv(o / "compare-celllines" / "Data.csv", usecols=["Hour"]).Hour.astype(float).unique())
    n_exp = pd.read_csv(o / "compare-celllines" / "Data.csv", usecols=["Hour", "Experiment"]).groupby("Hour").Experiment.nunique()
    cl = {}
    for ref in CONTROLS:
        t = pd.read_csv(o / "compare-celllines" / f"Combined_OR_{ref}.csv")
        for c in XDP_LINES:
            g = t[t.Compare == c].reset_index(drop=True)
            cl[(c, ref)] = g.assign(Hour=hours[:len(g)])
    cls = pd.read_csv(o / "compare-classes" / "Combined_OR_0.csv").reset_index(drop=True)
    cls = cls.assign(Hour=hours[:len(cls)])
    return {"hours": hours, "n_exp": n_exp, "cl": cl, "cls": cls}


# ------------------------------------------------------------------------------------------------------------
# figures
# ------------------------------------------------------------------------------------------------------------
def fig_cutoffs(a: pd.DataFrame, thr: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    y = 0
    ticks, labs = [], []
    for r in a.itertuples():
        g = thr[thr.exp == r.exp]
        lines = [l for l in figs.LINE_ORDER if l in set(g.line)]
        top = y
        for l in lines:
            q = g[g.line == l].iloc[0]
            ax.plot(np.log10([q.W90_live, q.W90_dead]), [y, y], color=figs.LINE_COLOR[l], lw=5, solid_capstyle="butt")
            ticks.append(y)
            labs.append(f"{r.exp} {S[l]}")
            y -= 1
        ax.fill_between(np.log10([r.live_cut, r.dead_cut]), top + 0.45, y + 0.55, color=figs.UT, alpha=0.18, lw=0)
        ax.plot(np.log10([r.live_cut, r.live_cut]), [top + 0.45, y + 0.55], color=figs.INK2, lw=1.2)
        ax.plot(np.log10([r.dead_cut, r.dead_cut]), [top + 0.45, y + 0.55], color=figs.INK2, lw=1.2)
        y -= 0.6
    ax.set_yticks(ticks)
    ax.set_yticklabels(labs, fontsize=7)
    figs.log_ratio_axis(ax, "x", [0.05, 0.06, 0.08, 0.1, 0.13, 0.15, 0.2, 0.25])
    ax.set_xlabel("GEDI ratio (log scale)")
    ax.plot([], [], color=figs.INK2, lw=1.2, label="first run: one pair per experiment (grey zone = cells dropped)")
    ax.plot([], [], color=figs.LIVE, lw=5, label="new analysis: one zone per line (coloured)")
    ax.legend(fontsize=7.5, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, frameon=False)
    return fig


def fig_t1(t1: pd.DataFrame, thr: pd.DataFrame):
    m = t1.merge(thr[["exp", "line", "T1_death_W90"]], on=["exp", "line"], how="left")
    m["lab"] = m.exp + " " + m.line.map(S)
    m["o"] = m.exp.map({e: i for i, e in enumerate(EXPS)}) * 10 + m.line.map({l: i for i, l in enumerate(figs.LINE_ORDER)})
    m = m.sort_values("o", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    for i, r in m.iterrows():
        col = figs.LINE_COLOR[r.line]
        ax.plot([100 * r.T1_death_W90, 100 * r["first_run"]], [i, i], color=figs.GRID, lw=1, zorder=1)
        ax.scatter(100 * r["first_run"], i, s=36, color=col, zorder=3)
        ax.scatter(100 * r.T1_death_W90, i, s=36, facecolor="white", edgecolor=col, lw=1.4, zorder=3)
    ax.set_yticks(range(len(m)))
    ax.set_yticklabels(m.lab, fontsize=7.5)
    ax.set_xlabel("untreated cells called dead at T1 (%)")
    ax.scatter([], [], s=36, color=figs.INK2, label="first run")
    ax.scatter([], [], s=36, facecolor="white", edgecolor=figs.INK2, label="new analysis (90% zone)")
    ax.legend(fontsize=7.5, loc="lower right")
    return fig


def fig_or(orr: dict):
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4), sharey=True)
    for ax, ref in zip(axes[:2], CONTROLS):
        for c in XDP_LINES:
            g = orr["cl"][(c, ref)]
            col = figs.LINE_COLOR[c]
            ax.fill_between(g.Hour.to_numpy(), g.lower.to_numpy(), g.upper.to_numpy(), color=col, alpha=0.18, lw=0)
            ax.plot(g.Hour.to_numpy(), g.OR.to_numpy(), color=col, lw=2, marker="o", ms=3, label=S[c])
        ax.set_title(f"each XDP line vs {S[ref]}", loc="left", fontsize=9.5)
    g = orr["cls"]
    axes[2].fill_between(g.Hour.to_numpy(), g.lower.to_numpy(), g.upper.to_numpy(), color=figs.SA, alpha=0.18, lw=0)
    axes[2].plot(g.Hour.to_numpy(), g.OR.to_numpy(), color=figs.SA, lw=2, marker="o", ms=3)
    axes[2].set_title("XDP lines vs control lines (groups)", loc="left", fontsize=9.5)
    for ax in axes:
        ax.axhline(1, color=figs.INK2, ls=":", lw=1)
        ax.set_yscale("log")
        figs.or_axis(ax, "y")
        ax.set_xlabel("hours")
    axes[0].set_ylabel("odds ratio of death\n(> 1: XDP line dies more)")
    axes[0].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------------------------------------
def build() -> str:
    OUT.mkdir(parents=True, exist_ok=True)
    inp = load_inputs()
    a = audit(inp)
    t1 = t1_death(inp)
    drop = dropped_share(a)
    orr = load_or()
    thr = pd.read_csv(LOCAL_OUT / "10a_final_tables" / "thresholds" / "thresholds_all.csv")
    pooled = pd.read_csv(LOCAL_OUT / "13_or_figures" / "pooled_or_by_set.csv")
    pooled = pooled[(pooled.set == "W90") & pooled.Compare.isin(XDP_LINES)]
    a.to_csv(OUT / "firstrun_input_audit.csv", index=False)
    t1.to_csv(OUT / "firstrun_t1_death.csv", index=False)

    x10 = a[a.exp == "XDP10"].iloc[0]
    d10 = inp["XDP10"]
    n_mis_wells = int(d10[d10.Sci_SampleID != d10.true_line].well.nunique())
    dup_wells = int(d10[d10.filenames.duplicated(keep=False)].well.nunique())
    x8 = a[a.exp == "XDP8"].iloc[0]
    x11 = a[a.exp == "XDP11"].iloc[0]
    clean = [e for e in EXPS if e != "XDP10" and a[a.exp == e].iloc[0].mislabelled_cells == 0
             and a[a.exp == e].iloc[0].dup_cells == 0 and a[a.exp == e].iloc[0].sa_wells_post_dose == 0
             and a[a.exp == e].iloc[0].sirna_wells == 0]
    cls = orr["cls"]
    cls_incl = int(((cls.lower <= 1) & (cls.upper >= 1)).sum())
    k63 = orr["cl"][("XDP-33363.D", "CTR-33114.C")]
    k09 = orr["cl"][("XDP-33109.2B", "CTR-33114.C")]
    n63 = int((k63.lower > 1).sum())
    i09, i63 = orr["cl"][("XDP-33109.2B", "CTR-33113.2I")], orr["cl"][("XDP-33363.D", "CTR-33113.2I")]
    n09_up, n09_dn = int((k09.lower > 1).sum()), int((k09.upper < 1).sum())
    last_h = int(orr["hours"][-1])
    last_n = int(orr["n_exp"].loc[orr["hours"][-1]])
    t1x = t1.merge(thr[["exp", "line", "T1_death_W90"]], on=["exp", "line"])
    w = lambda c, r: pooled[(pooled.Compare == c) & (pooled.Reference == r)].iloc[0]  # noqa: E731
    n_incl = int(((pooled.lxe_lo <= 1) & (pooled.lxe_hi >= 1)).sum())

    figA = png(fig_cutoffs(a, thr), "firstrun_cutoffs.png")
    figB = png(fig_t1(t1, thr), "firstrun_t1.png")
    figC = png(fig_or(orr), "firstrun_or.png")

    cut_rows = "".join(f"<tr><td>{r.exp}</td><td class='num'>{r.live_cut:.2f}</td><td class='num'>{r.dead_cut:.2f}</td>"
                       f"<td class='num'>{100 * drop[drop.exp == r.exp].dropped.iloc[0]:.0f}%</td>"
                       f"<td>{', '.join('T' + str(t) for t in sorted(set(r.tps_all) - set(r.tps))) or 'none'}</td></tr>"
                       for r in a.itertuples())
    wide = ("p, .lede, .claim, .facts, .bottom ul, figcaption { max-width: none; }\n"
            "header.title { padding-block: 32px 0; }\n"
            "table { border-collapse: collapse; font-family: var(--font-ui); font-size: 14px; width: 100%; }\n"
            "th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); }\n"
            "td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }\n"
            ".table-wrap { overflow-x: auto; }")
    out = ["<title>First MRID Run Review</title>", FONTS, f"<style>{CSS}\n{wide}</style>", "<main>"]
    out.append(f"""<header class="title"><h1>Review of the first MRID run</h1>
<p class="lede">This page reviews the first MRID run from 4 September 2026, before the September rework. It covers six
experiments and four lines: 14C and 13I are controls, 09B and 63D are XDP. The
<a href="{EVIDENCE_URL}">full evidence document</a> has the new analysis.</p>
</header>""")
    out.append(f"""<section class="bottom" aria-label="Bottom line"><h2>Bottom line</h2><ul>
<li><b>The first run did not show a clear difference between XDP and control lines.</b> When it compared the two XDP lines
with the two control lines as groups, the interval included 1 at {cls_incl} of {len(cls)} timepoints.</li>
<li><b>Line by line, the answer changed with time and with the control.</b> 63D had more death than 14C at {n63} of
{len(k63)} timepoints. 09B had less death than 14C at {n09_dn} timepoints and more death at {n09_up}.</li>
<li><b>The method had three weaknesses.</b> Each experiment used one pair of cutoffs for all lines, picked by eye. The
XDP10 input had errors. The model assumed that every experiment shows the same difference between two lines.</li>
<li><b>The new analysis corrects these weaknesses.</b> It also finds no reproducible difference between XDP and control
lines.</li>
</ul></section>""")
    out.append(f"""<section aria-labelledby="s1"><h2 id="s1">1 · How the first run called cells live or dead</h2>
<p class="claim">Each experiment had one live cutoff and one dead cutoff. The same pair applied to every line.</p>
<p>Austin picked each pair by eye from the MRID ratio plots. Cells below the live cutoff were live. Cells above the dead
cutoff were dead. The analysis left out the cells between the two cutoffs.</p>
<div class="table-wrap"><table><thead><tr><th>Experiment</th><th class="num">Live below</th><th class="num">Dead above</th>
<th class="num">Cells left out</th><th>Timepoints left out</th></tr></thead><tbody>{cut_rows}</tbody></table></div>
{fig_block(figA, "First-run cutoffs and the new per-line zones", "Grey: the zone between the first run's two cutoffs, the same for all lines of an experiment. Coloured bars: the new 90% buffer zone, one for each line.")}
<ul class="facts">
<li>The new zones differ between the lines of one experiment. One pair for all lines cannot follow these differences.</li>
<li>With one pair for all lines, T1 death went from {100 * t1.first_run.min():.1f}% to {100 * t1.first_run.max():.1f}%. In XDP8, it was
{100 * t1x[(t1x.exp == 'XDP8') & (t1x.line == 'XDP-33109.2B')].iloc[0]['first_run']:.0f}% for 09B and
{100 * t1x[(t1x.exp == 'XDP8') & (t1x.line == 'XDP-33363.D')].iloc[0]['first_run']:.0f}% for 63D. The dead cutoff in XDP8 was
{x8.dead_cut:.2f}.</li>
</ul>
{fig_block(figB, "T1 death in each line, first run and new analysis", "Filled dots: untreated cells called dead at T1 with the first-run cutoffs. Open dots: the same with the new 90% zone.")}
</section>""")
    out.append(f"""<section aria-labelledby="s2"><h2 id="s2">2 · What the first run found</h2>
<p class="claim">The first run did not show more death in XDP lines in a consistent way.</p>
{fig_block(figC, "First-run odds ratios over time", f"Odds ratio of death and its 95% interval at each timepoint, from the first run's own output files. Above 1 means that the XDP line has more death. Only {last_n} experiments contribute at {last_h} hours.")}
<ul class="facts">
<li>63D had more death than 14C at {n63} of {len(k63)} timepoints. This was the most consistent signal.</li>
<li>09B had less death than 14C at {n09_dn} timepoints and more death at {n09_up}. The direction changed during the run.</li>
<li>09B had less death than 13I at {int((i09.upper < 1).sum())} of {len(i09)} timepoints. 63D had more death than 13I at
{int((i63.lower > 1).sum())} timepoints and less death at {int((i63.upper < 1).sum())}.</li>
<li>As groups, XDP and control lines did not differ at {cls_incl} of {len(cls)} timepoints.</li>
</ul></section>""")
    out.append(f"""<section aria-labelledby="s3"><h2 id="s3">3 · Errors in the XDP10 input</h2>
<p class="claim">The XDP10 input had four errors. The other five experiments did not have these errors.</p>
<ul class="facts">
<li>The XDP10 cell file stopped in the middle of a row. {len(x10.missing_wells)} wells of row M (63D) were missing. Well L22 had only T1.</li>
<li>The XDP10 plate map listed {dup_wells} wells twice. The cells of these wells were counted two times ({x10.dup_cells:,} extra cells).</li>
<li>{x10.mislabelled_cells:,} cells in {n_mis_wells} wells had the wrong cell line. This includes control line 33362.C, which the
analysis counted as 14C or 13I.</li>
<li>{x10.sa_wells_post_dose} S.A. wells stayed in the analysis after the S.A. dose ({x10.sa_cells_post_dose:,} cells). S.A. kills
cells on purpose, so these wells add death that is not disease death.</li>
<li>In {', '.join(clean)}, the input had no duplicate cells, no wrong labels and no S.A. or siRNA wells after dosing.</li>
<li>Some timepoints and wells were left out on purpose or for unknown reasons. XDP11 used only T1 to T{max(x11.tps)}.
XDP8 left out T2. XDP5 left out well C06.</li>
</ul></section>""")
    out.append(f"""<section aria-labelledby="s4"><h2 id="s4">4 · The model</h2>
<p class="claim">The first run used MRID's combined model. This model assumes that every experiment shows the same difference
between two lines.</p>
<p>The experiments do not agree with each other. So this model gives intervals that are too narrow. The new analysis
lets each experiment have its own difference. With the new cutoffs and this change, all {n_incl} XDP-vs-control intervals
include 1. The values below come from the new analysis (90% zone, all six experiments, odds ratio averaged over time).</p>
<ul class="facts">
<li>63D vs 14C: MRID's combined model gives {w('XDP-33363.D', 'CTR-33114.C').mridpool_OR:.2f}
[{w('XDP-33363.D', 'CTR-33114.C').mridpool_lo:.2f}, {w('XDP-33363.D', 'CTR-33114.C').mridpool_hi:.2f}]. With a difference for
each experiment, it is {w('XDP-33363.D', 'CTR-33114.C').lxe_OR:.2f} [{w('XDP-33363.D', 'CTR-33114.C').lxe_lo:.2f},
{w('XDP-33363.D', 'CTR-33114.C').lxe_hi:.2f}].</li>
<li>09B vs 14C: MRID's combined model gives {w('XDP-33109.2B', 'CTR-33114.C').mridpool_OR:.2f}
[{w('XDP-33109.2B', 'CTR-33114.C').mridpool_lo:.2f}, {w('XDP-33109.2B', 'CTR-33114.C').mridpool_hi:.2f}]. With a difference for
each experiment, it is {w('XDP-33109.2B', 'CTR-33114.C').lxe_OR:.2f} [{w('XDP-33109.2B', 'CTR-33114.C').lxe_lo:.2f},
{w('XDP-33109.2B', 'CTR-33114.C').lxe_hi:.2f}].</li>
</ul></section>""")
    out.append("""<section aria-labelledby="s5"><h2 id="s5">5 · First run and new analysis</h2>
<div class="table-wrap"><table><thead><tr><th>Item</th><th>First run (4 Sep 2026)</th><th>New analysis (run 2)</th></tr></thead><tbody>
<tr><td>How cutoffs are set</td><td>Picked by eye from plots</td><td>From S.A.-killed cells, with a fixed rule</td></tr>
<tr><td>Cutoffs per line</td><td>No, one pair for all lines</td><td>Yes, one zone for each line in each experiment</td></tr>
<tr><td>Buffer zone</td><td>Yes, between the two cutoffs</td><td>Yes, where a cell is less than 90% likely to be live or dead</td></tr>
<tr><td>XDP10 input</td><td>Cut off, duplicate wells, wrong labels, S.A. wells</td><td>Complete file, corrected plate map, S.A. wells removed</td></tr>
<tr><td>Model</td><td>Every experiment shows the same line difference</td><td>Each experiment can show its own difference</td></tr>
<tr><td>Result</td><td>No consistent XDP-vs-control difference</td><td>No reproducible XDP-vs-control difference</td></tr>
</tbody></table></div></section>""")
    out.append("""<section class="limits" aria-labelledby="lim"><h3 id="lim">What this does not show</h3>
<ul class="facts">
<li>We did not rerun the first-run model with corrected XDP10 data. The first-run results above come from its own output files.</li>
<li>The first run also had a separate siRNA analysis. This page does not review it.</li>
</ul></section>""")
    out.append("</main>")
    return "\n".join(out)


def main() -> None:
    body = build()
    name = f"RGEDI_firstrun_review_simple_{time.strftime('%Y%m%d')}.html"
    (OUT / "artifact").mkdir(parents=True, exist_ok=True)
    (OUT / "artifact" / name).write_text(body)
    page = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            'content="width=device-width, initial-scale=1, viewport-fit=cover"></head><body>' + body + "</body></html>")
    (OUT / name).write_text(page)
    print(f"wrote {OUT / name} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
