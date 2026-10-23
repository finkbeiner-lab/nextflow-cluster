"""Step 10: build the ordered, self-contained HTML evidence document from the step outputs.

Every number in the document is read from a step output (summary.json / CSV); nothing is typed by
hand. Writes two files into <out>/docs/:
  RGEDI_SA_thresholds_evidence_<date>_<label>.html     standalone (open locally)
  artifact/<same name>.html                            same content without the document wrapper
                                                       (for publishing as a private claude.ai Artifact)
Usage: python -I steps/10_build_report.py <label e.g. CPA-draft>
"""

import html
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from scipy.stats import norm  # noqa: E402

from lib import cfg, figs  # noqa: E402
from lib import sets as SETS_  # noqa: E402
from lib import report as R  # noqa: E402
from lib.paths import CONFIG_DIR, LOCAL_OUT, RAW_CACHE, RUN  # noqa: E402
from lib.provenance import git_state, sha256_file, versions  # noqa: E402
from lib.stats import dl_pool  # noqa: E402

E = html.escape
P = SETS_.PRIMARY                                   # primary threshold set of this run (D28: W90)
SHORT_SET = {"W90": "90% window", "W95": "95% window", "D_sd": "single rule D cutoff", "B": "rule B",
             "C_sd": "rule C", "D_abs": "single cutoff, absolute transfer"}
PL = SHORT_SET.get(P, P)
SHORT = figs.LINE_SHORT
EXPS = figs.EXP_ORDER
STEPS = ["01_metadata", "01b_extract", "02a_xdp10_verify", "02c_crosscheck", "03_qc_counts", "04_dose_timing",
         "05_prepost", "05b_model_check", "05c_cutoff_preview", "05d_drift_preview", "06_fit_thresholds", "06b_window_cutoffs",
         "07_validity", "08_transfer", "09_sensitivity", "10a_final_tables", "11_mrid_inputs", "13_or_figures",
         "14_count_tables", "15_sa_capture", "16b_retention_fe", "17_count_synthesis"]

CSS = """
/* Layout: an analysis dossier - one reading column of numbered steps (a real sequence), a sticky step
   index on wide screens, and figures mounted on light specimen plates in both themes. */
:root {
  --bg: #f5f6f8; --surface: #ffffff; --fg: #16181d; --muted: #5a6070; --rule: #dcdfe6;
  --accent: #4a3aa7; --accent-soft: #ecebf8; --plate: #fcfcfb; --plate-ring: rgba(11,11,11,0.10);
  --ok: #1d6f31; --ok-bg: #e3f2e6; --warn: #7d5200; --warn-bg: #fbefd6; --fail: #a8231c; --fail-bg: #fbe4e2;
  --well-empty: #e8eaef; --l14c: #2a78d6; --l13i: #eb6834; --l09b: #1baf7a; --l63d: #eda100; --lout: #b8bcc6;
  --font-display: "IBM Plex Sans Condensed", "Arial Narrow", system-ui, sans-serif;
  --font-body: "Source Serif 4", Georgia, "Times New Roman", serif;
  --font-ui: "IBM Plex Sans", system-ui, -apple-system, "Segoe UI", sans-serif;
  --font-mono: "IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) {
  --bg: #111318; --surface: #181b22; --fg: #e7e9ef; --muted: #a3a9b8; --rule: #2c303b;
  --accent: #a99cf2; --accent-soft: #252040; --plate-ring: rgba(255,255,255,0.14);
  --ok: #7fd394; --ok-bg: #15301d; --warn: #f2c56b; --warn-bg: #33280f; --fail: #ff9a90; --fail-bg: #3b1714;
  --well-empty: #262a33; color-scheme: dark; } }
:root[data-theme="dark"] {
  --bg: #111318; --surface: #181b22; --fg: #e7e9ef; --muted: #a3a9b8; --rule: #2c303b;
  --accent: #a99cf2; --accent-soft: #252040; --plate-ring: rgba(255,255,255,0.14);
  --ok: #7fd394; --ok-bg: #15301d; --warn: #f2c56b; --warn-bg: #33280f; --fail: #ff9a90; --fail-bg: #3b1714;
  --well-empty: #262a33; color-scheme: dark; }
body { background: var(--bg); color: var(--fg); font-family: var(--font-body); font-size: 16.5px; line-height: 1.6;
  padding-inline: 16px; padding-block: 0 48px; }
.page { max-width: 1240px; margin: 0 auto; display: grid; grid-template-columns: minmax(0, 1fr); gap: 40px; }
@media (min-width: 1100px) { .page { grid-template-columns: 220px minmax(0, 1fr); } }
nav.toc { display: none; }
@media (min-width: 1100px) { nav.toc { display: block; position: sticky; top: env(safe-area-inset-top, 0px);
  align-self: start; padding-top: 32px; font-family: var(--font-ui); font-size: 13px; max-height: 100vh; overflow-y: auto; } }
nav.toc ol { list-style: none; margin: 0; padding: 0; display: grid; gap: 4px; }
nav.toc a { color: var(--muted); text-decoration: none; display: block; padding: 3px 8px; border-radius: 4px; }
nav.toc a:hover, nav.toc a:focus-visible { color: var(--fg); background: var(--accent-soft); outline: none; }
main { min-width: 0; max-width: 900px; }
header.title { padding-block: 40px 24px; border-bottom: 1px solid var(--rule); display: grid; gap: 10px; }
.eyebrow { font-family: var(--font-ui); font-size: 12.5px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--accent); font-weight: 600; }
h1, h2, h3, h4 { font-family: var(--font-display); text-wrap: balance; line-height: 1.2; margin: 0; }
h1 { font-size: clamp(30px, 4.6vw, 44px); font-weight: 600; letter-spacing: -0.01em; }
h2 { font-size: 27px; font-weight: 600; }
h3 { font-size: 20px; font-weight: 600; margin-top: 8px; }
h4 { font-size: 16px; font-weight: 600; font-family: var(--font-ui); }
p, li { max-width: 70ch; }
.lede { font-size: 19px; color: var(--muted); max-width: 64ch; }
.meta { font-family: var(--font-ui); font-size: 13.5px; color: var(--muted); display: flex; flex-wrap: wrap; gap: 6px 18px; }
section.step { padding-block: 36px 8px; border-bottom: 1px solid var(--rule); display: grid; gap: 14px; }
.step-head { display: grid; grid-template-columns: auto minmax(0, 1fr); gap: 4px 14px; align-items: baseline; }
.step-no { font-family: var(--font-mono); font-size: 14px; color: var(--accent); border: 1px solid var(--accent);
  border-radius: 4px; padding: 1px 7px; }
.qmed { display: grid; gap: 10px; }
.box { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 14px 18px; display: grid; gap: 6px; }
.box > .k { font-family: var(--font-ui); font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); font-weight: 600; }
.box.takeaway { border-left: 3px solid var(--accent); }
.box.worry { background: transparent; border-style: dashed; }
.box p { margin: 0; }
figure { margin: 6px 0 4px; display: grid; gap: 6px; }
img.plate { max-width: 100%; height: auto; background: var(--plate); border-radius: 6px; box-shadow: 0 0 0 1px var(--plate-ring); }
figcaption { font-family: var(--font-ui); font-size: 13.5px; color: var(--muted); max-width: 80ch; }
.fig-no { font-weight: 600; color: var(--fg); }
.table-wrap { overflow-x: auto; max-width: 100%; }
table { border-collapse: collapse; font-family: var(--font-ui); font-size: 13.5px; min-width: 100%; }
th, td { text-align: left; padding: 6px 10px; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; color: var(--muted); font-size: 12px; letter-spacing: 0.04em; text-transform: uppercase; white-space: nowrap; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
code, .mono { font-family: var(--font-mono); font-size: 0.88em; }
.badge { font-family: var(--font-ui); font-size: 11.5px; font-weight: 600; padding: 2px 8px; border-radius: 999px; white-space: nowrap; display: inline-block; }
.badge.ok { color: var(--ok); background: var(--ok-bg); } .badge.warn { color: var(--warn); background: var(--warn-bg); }
.badge.fail { color: var(--fail); background: var(--fail-bg); } .badge.muted { color: var(--muted); background: var(--well-empty); }
.tag { font-family: var(--font-ui); font-size: 11px; font-weight: 600; letter-spacing: 0.04em; padding: 1px 6px; border-radius: 3px; border: 1px solid currentColor; }
.tag.verified { color: var(--ok); } .tag.inferred { color: var(--warn); } .tag.choice { color: var(--accent); }
details { background: var(--surface); border: 1px solid var(--rule); border-radius: 6px; padding: 10px 16px; }
details > summary { cursor: pointer; font-family: var(--font-ui); font-weight: 600; font-size: 14.5px; }
details[open] > summary { margin-bottom: 10px; }
details .inner { display: grid; gap: 12px; }
.grid2 { display: grid; gap: 16px; grid-template-columns: repeat(auto-fit, minmax(min(100%, 380px), 1fr)); }
.plate-map { display: grid; grid-template-columns: 18px repeat(24, minmax(10px, 1fr)); gap: 2px; font-family: var(--font-mono);
  font-size: 9px; color: var(--muted); min-width: 420px; }
.plate-map .w { aspect-ratio: 1; border-radius: 50%; background: var(--well-empty); }
.plate-map .lab { display: flex; align-items: center; justify-content: center; }
.w.l14c { background: var(--l14c); } .w.l13i { background: var(--l13i); } .w.l09b { background: var(--l09b); }
.w.l63d { background: var(--l63d); } .w.lout { background: var(--lout); }
.w.sirna { background-image: repeating-linear-gradient(45deg, rgba(255,255,255,0.75) 0 2px, transparent 2px 4px); }
.w.sa { box-shadow: 0 0 0 2px var(--accent); }
.legend { display: flex; flex-wrap: wrap; gap: 6px 16px; font-family: var(--font-ui); font-size: 12.5px; color: var(--muted); align-items: center; }
.legend .sw { display: inline-block; width: 11px; height: 11px; border-radius: 50%; vertical-align: -1px; margin-right: 5px; }
.summary-list { display: grid; gap: 8px; padding-left: 20px; margin: 0; }
.decide { border: 1px solid var(--warn); background: var(--warn-bg); border-radius: 6px; padding: 14px 18px; display: grid; gap: 8px; }
.decide h4 { color: var(--warn); }
a { color: var(--accent); }
:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
footer { font-family: var(--font-ui); font-size: 12.5px; color: var(--muted); padding-block: 24px; display: grid; gap: 8px; }
@media (prefers-reduced-motion: reduce) { * { scroll-behavior: auto !important; } }
@media print { nav.toc { display: none; } details { border: none; } }
"""

FONTS = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" '
         'href="https://fonts.gstatic.com" crossorigin><link rel="stylesheet" href="https://fonts.googleapis.com/css2?'
         'family=IBM+Plex+Mono:wght@400;500&family=IBM+Plex+Sans+Condensed:wght@500;600&family=IBM+Plex+Sans:'
         'wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap">')


def tag(kind: str) -> str:
    return f'<span class="tag {kind}">{kind}</span>'


def step_block(sid: str, no: str, title: str, question: str, method: str, evidence: str, takeaway: str,
               decisions: str = "", worry: str = "") -> str:
    parts = [f'<section class="step" id="{sid}"><div class="step-head"><span class="step-no">{E(no)}</span>'
             f'<h2>{E(title)}</h2></div>',
             f'<div class="box"><span class="k">Question</span><p>{question}</p></div>',
             f'<div class="qmed"><span class="eyebrow">Method</span>{method}</div>',
             f'<div class="qmed"><span class="eyebrow">Evidence</span>{evidence}</div>',
             f'<div class="box takeaway"><span class="k">Takeaway</span>{takeaway}</div>']
    if decisions:
        parts.append(f'<div class="qmed"><span class="eyebrow">Decisions</span>{decisions}</div>')
    if worry:
        parts.append(f'<div class="box worry"><span class="k">What would have worried us</span>{worry}</div>')
    parts.append("</section>")
    return "\n".join(parts)


def decision_refs(ids, decs) -> str:
    out = []
    for d in decs:
        if d["id"] in ids:
            out.append(f'<p><a href="#{d["id"]}"><b>{d["id"]}</b></a> {R.badge(d["status"])} {E(str(d["question"]))}</p>')
    return "".join(out)


# ----------------------------------------------------------------------------------------------------------
def plate_map(exp: str) -> str:
    roles = cfg.well_roles(exp)
    cls_line = {"CTR-33114.C": "l14c", "CTR-33113.2I": "l13i", "XDP-33109.2B": "l09b", "XDP-33363.D": "l63d"}
    by = {r.well: r for r in roles.itertuples()}
    cells = ['<span class="lab"></span>'] + [f'<span class="lab">{c}</span>' for c in range(1, 25)]
    for row in "ABCDEFGHIJKLMNOP":
        cells.append(f'<span class="lab">{row}</span>')
        for c in range(1, 25):
            w = f"{row}{c:02d}"
            r = by.get(w)
            if r is None:
                cells.append('<span class="w" title="' + w + ': not used"></span>')
                continue
            k = ["w", cls_line.get(r.line, "lout")]
            if r.condition == "sirna":
                k.append("sirna")
            if r.condition == "sa":
                k.append("sa")
            title = f"{w}: {r.line} - " + {"sa": "S.A.", "untreated": "untreated", "sirna": str(r.sirna_name)}[r.condition]
            cells.append(f'<span class="{" ".join(k)}" title="{E(title)}"></span>')
    return f'<div class="table-wrap"><div class="plate-map" role="img" aria-label="{exp} plate map">{"".join(cells)}</div></div>'


PLATE_LEGEND = ('<div class="legend"><span><span class="sw" style="background:var(--l14c)"></span>14C</span>'
                '<span><span class="sw" style="background:var(--l13i)"></span>13I</span>'
                '<span><span class="sw" style="background:var(--l09b)"></span>09B</span>'
                '<span><span class="sw" style="background:var(--l63d)"></span>63D</span>'
                '<span><span class="sw" style="background:var(--lout)"></span>out of scope (33362.C)</span>'
                '<span><span class="sw" style="background:var(--well-empty);box-shadow:0 0 0 2px var(--accent)"></span>'
                'S.A. well (ring)</span><span><span class="sw" style="background:repeating-linear-gradient(45deg,'
                'var(--muted) 0 2px,transparent 2px 4px)"></span>siRNA (hatched, excluded)</span></div>')


# ----------------------------------------------------------------------------------------------------------
def ci(v, lo, hi, nd=3, mark=""):
    if v is None or pd.isna(v):
        return "–"
    if lo is None or pd.isna(lo):
        return f"{v:.{nd}f}"
    return f"{v:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]{mark}"


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    o = df.exp.map({e: i for i, e in enumerate(EXPS)}) * 10 + df.line.map({l: i for i, l in enumerate(figs.LINE_ORDER)})
    return df.assign(_o=o).sort_values("_o").drop(columns="_o")


FEW = " †"


def step_6b(decs, F) -> str:
    """Run 2 (D28): the buffer zone around the rule D centre."""
    s6b = R.step_json("06b_window_cutoffs", "summary")
    if not s6b:
        return ""
    from scipy.stats import norm
    d = R.step_csv("06_fit_thresholds", "thresholds_direct.csv")
    u = ordered(d[d.usable])
    death = R.step_csv("07_validity", "reference_death_by_tp.csv")
    t1 = death[death.tp == 1]
    s08 = R.step_json("08_transfer", "summary")
    called = {k: 1 - norm.cdf((np.log10(u[c]) - u.dead_mu_log) / u.dead_sd) for k, c in
              (("D", "D_live"), ("W90", "W90_dead"), ("W95", "W95_dead"), ("B", "B_dead"))}
    drop90 = t1[t1.rule == "W90"].ambiguous
    rows = [{"exp": r.exp, "line": SHORT[r.line], "c": f"{r.D_live:.3f}",
             "w90": f"{r.W90_live:.3f}–{r.W90_dead:.3f}", "f90": f"{r.W90_dead / r.W90_live:.2f}×",
             "w95": f"{r.W95_live:.3f}–{r.W95_dead:.3f}" + (" *" if str(r.W95_dead_capped) == "True" else "")}
            for r in u.itertuples()]
    f6 = F.get("06b_window_cutoffs", {})
    ev = f"""{R.figure(f6['F6W'], 'Figure 6b.1') if 'F6W' in f6 else ''}
{R.table(pd.DataFrame(rows), ["exp", "line", "c", "w90", "f90", "w95"],
         ["Exp", "Line", "Rule D centre", "90% window: live below – dead above", "Width", "95% window"],
         num_cols=["c", "f90"])}
<p>* Dead edge capped at the centre of the S.A.-dead population (the 95% level is never reached there). Window edges for the
cases without their own S.A. fit are transferred like the other cutoffs (live-SD offsets, Step 8); leave-one-experiment-out
error of the transferred edges: median {s08['loeo_abs_err_median']['W90_live']:.2f} (live edge) and
{s08['loeo_abs_err_median']['W90_dead']:.2f} (dead edge) log10 units, against {s08['loeo_abs_err_median']['D_live']:.2f} for the
single cutoff.</p>
<p>What the window costs and buys, on the 90% level: {100 * drop90.median():.1f}% of untreated cells at T1 fall in the zone and
are dropped (median; up to {100 * drop90.max():.1f}%). Of the S.A.-dead population, the 90% window still calls a median
{100 * np.median(called['W90']):.0f}% dead (95% window {100 * np.median(called['W95']):.0f}%), against
{100 * np.median(called['D']):.0f}% for the single cutoff and {100 * np.median(called['B']):.0f}% for rule B, whose dead cutoff sits
much higher.</p>"""
    take = (f"<p>The window is centred where a cell is equally likely to be live or dead and is {rng_txt(*s6b['W90']['fold_width_range'])}× "
            "wide at 90% (median "
            f"{s6b['W90']['fold_width_median']:.2f}×): narrow where S.A. separates the two populations well (XDP1, XDP5) and wide "
            "where they overlap (XDP11). It drops a few percent of untreated cells and keeps most S.A.-killed cells in the dead "
            "class, unlike rule B. From Step 11 on, this 90% window is the primary threshold set of run 2; the single cutoff of run 1 "
            "and the 95% window are shown as comparisons.</p>")
    return step_block(
        "s6b", "6b", "Buffer zone around the cutoff (run 2)",
        "Rule D gives one cutoff, so every cell is called live or dead. Where should a buffer zone of undecided cells go, and "
        "how wide should it be?",
        "<p>Centre: rule D, where the untreated live population and the S.A.-dead population are equally dense. Edges: where "
        "a cell becomes 9 times (90% window) or 19 times (95% window) more likely to belong to one population than the other. "
        "The same two fitted populations set the centre and the edges, so the width adapts to how well they separate in each "
        "line × experiment (D28).</p>", ev, take, decision_refs(["D28"], decs))


def steps_6_to_10(decs, params, F, S) -> str:
    out = []
    tol = params["transfer"]["loeo_tolerance_log10"]
    d6 = R.step_csv("06_fit_thresholds", "thresholds_direct.csv")
    if not len(d6):
        return ""
    d6 = ordered(d6)
    s6, s7, s10 = (S.get(k, {}) for k in ("06_fit_thresholds", "07_validity", "10a_final_tables"))
    k_sd = params["cutoff_rule"]["k_sd"]

    # ---- Step 6 ----------------------------------------------------------------------------------------
    tests = ["U_cells", "U_wells", "U_dead_found", "U_comp_cells", "U_bic", "U_separation", "U_dead_excess"]
    t_lab = {"U_cells": "≥500 cells", "U_wells": "≥2 wells", "U_dead_found": "no dead component", "U_comp_cells": "≥50/comp.",
             "U_bic": "BIC gain", "U_separation": "D < 2", "U_dead_excess": "dead share < reference + 10 points"}
    rows = []
    for r in d6.itertuples():
        fails = [t_lab[t] for t in tests if not getattr(r, t)]
        mk = FEW if r.few_wells else ""
        rows.append({"exp": r.exp, "line": SHORT[r.line], "fit": "T" + str(r.fit_tps).replace(",", ",T"),
                     "use": R.badge("PASS" if r.usable else "WARN", "usable" if r.usable else "transfer: " + "; ".join(fails)),
                     "ref": f"{r.mode_ratio:.3f} (SD {r.sd_inner:.2f})",
                     "dead": f"{r.dead_mu_ratio:.2f}; {100 * r.dead_w:.1f}% vs {100 * r.ref_dead_w:.1f}%" if r.dead_found else "–",
                     "D": ci(r.D_live, getattr(r, "D_live_lo", np.nan), getattr(r, "D_live_hi", np.nan), mark=mk) if r.dead_found else "–",
                     "C": ci(r.C_live, getattr(r, "C_live_lo", np.nan), getattr(r, "C_live_hi", np.nan), mark=mk) if r.dead_found else "–",
                     "B": (f"{r.B_live:.3f}–{r.B_dead:.3f}" if r.dead_found else "–"),
                     "miss": (f"{100 * norm.cdf((np.log10(r.B_dead) - r.dead_mu_log) / r.dead_sd):.0f}%" if r.dead_found else "–")})
    figs6 = "".join(f'<details><summary>{e}: cutoffs on the distributions</summary><div class="inner">{R.figure(F["06_fit_thresholds"][f"F6_{e}"])}</div></details>'
                    for e in ["XDP11", "XDP10", "XDP5", "XDP1", "XDP3"] if f"F6_{e}" in F["06_fit_thresholds"])
    u6 = d6[d6.usable].copy()
    u6["B_miss"] = norm.cdf((np.log10(u6.B_dead) - u6.dead_mu_log) / u6.dead_sd)
    u6["D_width"] = u6.D_live_hi - u6.D_live_lo
    spread = u6.groupby("exp").agg(lo=("D_live", "min"), hi=("D_live", "max"), w=("D_width", "median"), n=("D_live", "size"))
    spread = spread[spread.n > 1]
    alt = u6[u6.get("boot_alt_solution_frac", pd.Series(0, index=u6.index)).fillna(0) > 0.1] if "boot_alt_solution_frac" in u6 else u6.iloc[0:0]
    ev6 = f"""{R.table(pd.DataFrame(rows), ["exp", "line", "fit", "use", "ref", "dead", "D", "C", "B", "miss"],
        ["Exp", "Line", "S.A. readout", "Usability", "Reference live peak (log SD)", "S.A. dead: centre; share in S.A. vs reference",
         "Rule D cutoff [95% CI]", "Rule C cutoff [95% CI]", "Rule B live–dead", "S.A.-dead pop. below rule B dead cutoff"],
        raw_cols=["use"])}
<p>Intervals: wells redrawn {s6.get('B', '?')} times.{'' if 'seed_scheme' in s6 else ' <b>These intervals come from a run whose random seeds were not fixed; a rerun with fixed seeds is in progress (point estimates are unaffected).</b>'} † XDP10 has 2 S.A. wells per line: its cells are redrawn within those
two wells, so its intervals understate the uncertainty. "Share in S.A. vs reference" is the dead component's weight in
post-dose S.A. cells and in the reference cells, with the components held fixed.</p>{figs6}"""
    out.append(step_block(
        "s6", "6", "Cutoffs fitted on S.A.-killed cells",
        "For each line in each experiment with a usable S.A. dead population, where do the rule D, C and B cutoffs fall, "
        "and how precisely are they determined?",
        "<p>Per line and experiment: the reference live population (peak and spread below the peak) from untreated wells at "
        "the S.A. readout (XDP1: the same wells at T10); the S.A. dead population from a 3-component mixture on post-dose "
        "S.A. cells (D18); then rules B, C and D (D20). The usability tests fixed in Step 5 decide which cases keep their "
        "own cutoffs.</p>", ev6,
        f"<p>{len(u6)} of {len(d6)} S.A. cases are usable. Rule D cutoffs fall at {rng_txt(u6.D_live.min(), u6.D_live.max(), 3)} "
        f"and rule C at {rng_txt(u6.C_live.min(), u6.C_live.max(), 3)}; rule B drops cells between "
        f"{rng_txt(u6.B_live.min(), u6.B_live.max(), 3)} and {rng_txt(u6.B_dead.min(), u6.B_dead.max(), 3)}, and its dead cutoff "
        f"leaves a median {100 * u6.B_miss.median():.0f}% (range {100 * u6.B_miss.min():.0f}–{100 * u6.B_miss.max():.0f}%) of the "
        f"S.A.-dead population uncalled. Within an experiment the rule-D cutoffs of different lines differ by "
        + "; ".join(f"{e} {r.lo:.3f}–{r.hi:.3f}" for e, r in spread.iterrows())
        + f", about the width of a single case's interval (median {rng_txt(spread.w.min(), spread.w.max(), 3)}): per-line "
        "cutoffs are not clearly different from one another within an experiment."
        + (f" In {', '.join(a.exp + ' ' + SHORT[a.line] for a in alt.itertuples())} a share of bootstrap fits settled on a "
           "second mixture solution, which widens its interval." if len(alt) else "") + "</p>",
        decision_refs(["D18", "D20"], decs),
        "<p>Cutoffs that varied widely between lines of one experiment, or intervals as wide as those differences. The second "
        "happened: within-experiment line differences are of the same size as the intervals.</p>"))
    out.append(step_6b(decs, F))

    # ---- Step 7 ----------------------------------------------------------------------------------------
    tiers = ordered(R.step_csv("07_validity", "drift_tiers.csv"))
    edge = R.step_csv("07_validity", "edge_xdp10.csv")
    nat = ordered(R.step_csv("07_validity", "natural_vs_sa.csv"))
    death = R.step_csv("07_validity", "reference_death_by_tp.csv")
    tiers["badge"] = tiers.tier.map(lambda t: R.badge({"STABLE": "PASS", "MODERATE": "WARN", "MAJOR": "FAIL"}[t], t))
    tiers["lab"] = tiers.exp + " " + tiers.line.map(SHORT)
    tiers["sus"] = tiers.sustained_drift_sd.map(lambda v: f"{v:.2f}")
    tiers["mx"] = [f"{a:.2f} (T{int(b)})" for a, b in zip(tiers.max_abs_drift_sd, tiers.max_at_tp)]
    dd = death[death.rule == "D"].sort_values("tp")
    dd["jump"] = dd.groupby(["exp", "line"]).death_frac.diff()
    stable_keys = set(zip(tiers[tiers.tier == "STABLE"].exp, tiers[tiers.tier == "STABLE"].line))
    jumps = dd[[k in stable_keys for k in zip(dd.exp, dd.line)] & (dd.jump > 0.08)]
    nat2 = nat.assign(lab=nat.exp + " " + nat.line.map(SHORT) + np.where(nat.exp == "XDP1", " ‡", ""),
                      sa=nat.sa_dead_median.map(lambda v: f"{v:.2f}" if pd.notna(v) else "–"),
                      ref=nat.ref_dead_median.map(lambda v: f"{v:.2f}" if pd.notna(v) else "–"),
                      surv=nat.survivor_shift_sd.map(lambda v: f"{v:+.2f}" if pd.notna(v) else "–"))
    both = nat.dropna(subset=["sa_dead_median", "ref_dead_median"])
    surv_hi = nat[nat.exp.isin(["XDP1", "XDP3", "XDP5"])].survivor_shift_sd
    surv_lo = nat[nat.exp.isin(["XDP10", "XDP11"])].survivor_shift_sd
    ev7 = f"""<h3>7.1 Drift of the untreated live population</h3>
{R.table(tiers, ["lab", "sus", "mx", "badge"], ["Case", "Drift reached at ≥ 2 timepoints (live SDs)", "Largest single value (timepoint)", "Tier"],
         num_cols=["sus"], raw_cols=["badge"])}
<p>The tier uses the drift reached at two or more timepoints, so a single imaging-day spike (e.g. XDP8 T2, XDP1 T7) does
not decide it; the largest single value is shown next to it.</p>
{R.figure(F['07_validity']['F8'], 'Figure 7.1') if 'F8' in F.get('07_validity', {}) else ''}
<h3>7.2 Edge column (XDP10)</h3>
{R.table(edge.assign(l=edge.line.map(SHORT), a=edge.peak_col22.map(lambda v: f"{v:.3f}"), b=edge.peak_col21.map(lambda v: f"{v:.3f}"),
                     c=edge.diff_sd.map(lambda v: f"{v:+.2f}")),
         ["l", "a", "b", "c"], ["Line", "Live peak, column 22 (S.A.), T1–T8", "Live peak, column 21 (untreated)", "Difference (live SDs)"],
         num_cols=["a", "b", "c"])}
<h3>7.3 Naturally dying vs S.A.-killed cells, and S.A. survivors</h3>
{R.table(nat2, ["lab", "sa", "ref", "n_ref_dead", "surv"],
         ["Case", "Median ratio, S.A. cells called dead (rule D)", "Median ratio, untreated cells called dead at the last image",
          "Untreated cells called dead", "S.A. survivors' peak vs untreated peak (live SDs)"], num_cols=["n_ref_dead"])}
<p>‡ XDP1 has no untreated wells: its "untreated" columns are all wells at T10, before dosing.</p>"""
    out.append(step_block(
        "s7", "7", "Do the cutoffs hold?",
        "Is one fixed cutoff per line and experiment appropriate over the whole run, is XDP10's edge column a problem, and do "
        "naturally dying cells sit where S.A.-killed cells sit?",
        "<p>Using the final cutoffs (direct or transferred): the live peak of untreated wells at every timepoint, in units of "
        "the live SD (stable ≤ 0.5, moderate ≤ 1, major > 1, reached at two or more timepoints); XDP10's S.A. column vs its "
        "neighbour before dosing; and the ratio of cells called dead in untreated vs S.A. wells.</p>", ev7,
        f"<p>Drift: {s7.get('drift_tiers', {}).get('STABLE', 0)} stable, {s7.get('drift_tiers', {}).get('MODERATE', 0)} moderate, "
        f"{s7.get('drift_tiers', {}).get('MAJOR', 0)} major ({', '.join(s7.get('major_drift', [])) or 'none'}); for the major "
        "cases one fixed cutoff turns drift into apparent death or survival (D22). Even in stable cases the share called dead "
        + (f"jumps by more than 8 points from one image to the next in {len(jumps)} instances "
           f"({', '.join(sorted({f'{e} T{int(t)}' for e, t in zip(jumps.exp, jumps.tp)}))}), mostly on the plate-wide days in the ledger. "
           if len(jumps) else "has no jumps above 8 points between images. ")
        + "Before dosing, XDP10's column 22 matched column 21 in every line (≤ "
        f"{edge.diff_sd.abs().max():.2f} SD); an edge effect after dosing cannot be excluded. Untreated cells called dead sit at "
        f"a slightly lower ratio than S.A.-killed cells (median difference {(both.sa_dead_median - both.ref_dead_median).median():.2f}). "
        f"S.A. survivors sit {rng_txt(surv_hi.min(), surv_hi.max())} live SDs above the untreated peak in XDP1/3/5 and "
        f"{rng_txt(surv_lo.min(), surv_lo.max())} in XDP10/11, so survivors are stressed, not unchanged.</p>",
        decision_refs(["D22"], decs),
        "<p>Major drift in many cases, an edge effect in XDP10, or naturally dying cells at a very different ratio from "
        "S.A.-killed cells. Drift is limited (above); no edge effect before dosing; the two kinds of dead cells sit close.</p>"))

    # ---- Step 8 ----------------------------------------------------------------------------------------
    lo = R.step_csv("08_transfer", "loeo_validation.csv")
    tr = ordered(R.step_csv("08_transfer", "thresholds_transfer.csv"))
    lrows = []
    for c, name in (("D_live", "Rule D"), ("C_live", "Rule C (= rule B live cutoff)")):
        e_sd, e_ab, e_po = lo[f"{c}_err_log10"].abs(), lo[f"{c}_err_absolute"].abs(), lo[f"{c}_err_pooled"].abs()
        lrows.append({"rule": name, "sd": f"{e_sd.median():.3f} (max {e_sd.max():.3f})", "ok": f"{int((e_sd <= tol).sum())} of {len(lo)}",
                      "abs": f"{e_ab.median():.3f} (max {e_ab.max():.3f})", "oka": f"{int((e_ab <= tol).sum())} of {len(lo)}",
                      "pool": f"{e_po.median():.3f}"})
    trows = [{"exp": r.exp, "line": SHORT[r.line], "ref": r.reference, "src": f"{r.sources} ({r.n_sources})",
              "D": ci(r.D_live, r.D_live_lo, r.D_live_hi), "Da": f"{r.D_live_absolute:.3f}" if "D_live_absolute" in tr else "–",
              "C": ci(r.C_live, r.C_live_lo, r.C_live_hi), "B": f"{r.B_live:.3f}–{r.B_dead:.3f}"} for r in tr.itertuples()]
    ev8 = f"""{R.table(pd.DataFrame(lrows), ["rule", "sd", "ok", "abs", "oka", "pool"],
        ["Cutoff", "Live-SD offset: |error|, log10 (median, max)", f"Within ±{tol}", "Absolute log offset: |error| (median, max)",
         f"Within ±{tol}", "Live-SD offset pooled over lines: median |error|"])}
<p>Rule B's dead cutoff is the live peak + 2.5 live SDs in every source case, so it transfers exactly by construction and is
not tested here; rule B's live cutoff equals rule C.</p>
{R.figure(F['08_transfer']['F10'], 'Figure 8.1') if 'F10' in F.get('08_transfer', {}) else ''}
<h3>Transferred cutoffs</h3>
{R.table(pd.DataFrame(trows), ["exp", "line", "ref", "src", "D", "Da", "C", "B"],
         ["Exp", "Line", "Own live reference", "Source experiments (n)", "Rule D, live-SD offset [resampling interval]",
          "Rule D, absolute offset (D21 alternative)", "Rule C [resampling interval]", "Rule B live–dead"])}
<p>The intervals resample the source experiments and the target's wells only; they do not include the choice of offset
form or the leave-one-out error above.</p>"""
    lo_d = lo["D_live_err_log10"].abs()
    out.append(step_block(
        "s8", "8", "Transfer to cases without a usable S.A. dead population",
        "How are cutoffs set for XDP8 (no S.A. images) and the S.A. cases that failed usability, and how accurate is that?",
        "<p>Each usable case's cutoff is written as an offset from its own untreated live peak, in live-SD units (planned, D03) "
        "or in absolute log units (alternative); a target gets its own live peak plus the median offset of its line. Accuracy "
        "is tested by leaving one experiment out and predicting its directly fitted cutoff from the others.</p>", ev8,
        f"<p>With the planned live-SD offsets, held-out rule-D cutoffs are predicted with a median error of {lo_d.median():.3f} "
        f"log10 units (max {lo_d.max():.3f}; tolerance {tol}); absolute offsets do better for rules D and C (table). The offset "
        "form changes the transferred cutoffs substantially - most for XDP3, whose live population is narrow - so it needs a "
        "decision (D21). Transferred cutoffs are less certain than direct ones in either form.</p>",
        decision_refs(["D03", "D21"], decs),
        "<p>Large leave-one-out errors. They exceed the planned tolerance for the live-SD form.</p>"))

    # ---- Step 9 ----------------------------------------------------------------------------------------
    sens = R.step_csv("09_sensitivity", "sensitivity.csv")
    srows = []
    for v, g in (sens.groupby("variant", sort=False) if len(sens) else []):
        def mx_(c):
            return f"{g[c].abs().max():.3f}" if c in g and g[c].notna().any() else "–"
        def md_(c):
            return f"{100 * g[c].abs().max():.1f}" if c in g and g[c].notna().any() else "–"
        srows.append({"v": v, "n": len(g), "D": mx_("D_dlog_live"), "C": mx_("C_dlog_live"), "Bd": mx_("B_dlog_dead"),
                      "dD": md_("D_ddeath"), "dC": md_("C_ddeath"), "lost": int((~g.dead_found.astype(bool)).sum())})
    ev9 = "<p>Not yet run.</p>" if not srows else R.table(pd.DataFrame(srows), ["v", "n", "D", "C", "Bd", "dD", "dC", "lost"],
        ["Variant", "Cases", "Largest change, rule D (log10)", "… rule C", "… rule B dead", "Largest change in untreated called dead, D (points)",
         "… C (points)", "Dead population lost"], num_cols=["n", "D", "C", "Bd", "dD", "dC", "lost"])
    out.append(step_block(
        "s9", "9", "Sensitivity analyses",
        "How much do the cutoffs move under reasonable alternative choices?",
        "<p>Point estimates for every usable case under: 2.0 and 3.0 SD instead of 2.5 (rules B and C; rule D does not use an "
        "SD multiplier); single fit timepoints (XDP10 T9 or T10, XDP1 T13 or T14) and XDP5 T8+T9; XDP11 without the column-22 "
        "S.A. wells; and the S.A. dead population pooled over all lines of an experiment.</p>", ev9,
        "<p>Each entry is the largest change over the usable cases. Rule D moves little with the fit timepoint or the edge "
        "column; rule C moves with the SD multiplier by design; pooling the dead population over lines moves rule D by at most "
        "the value shown.</p>"))

    # ---- Step 10 ---------------------------------------------------------------------------------------
    fa = ordered(R.step_csv("10a_final_tables", "thresholds/thresholds_all.csv"))
    frows = [{"exp": r.exp, "line": SHORT[r.line], "src": "direct" if r.source.startswith("direct") else "transfer",
              "D": ci(r.D_live, r.D_live_lo, r.D_live_hi, mark=FEW if str(r.few_wells) == "True" else ""), "C": f"{r.C_live:.3f}",
              "W": f"{r.W90_live:.3f}–{r.W90_dead:.3f}",
              "B": f"{r.B_live:.3f}–{r.B_dead:.3f}",
              "old": f"{r.firstrun_live:g}/{r.firstrun_dead:g} · {r.sept_live:.3f}/{r.sept_dead:.3f}",
              "t1": f"{100 * r.T1_death_W90:.1f} / {100 * r.T1_death_D:.1f} / {100 * r.T1_death_C:.1f} / {100 * r.T1_death_B:.1f}",
              "drift": r.tier} for r in fa.itertuples()]
    t1 = s10.get("T1_death_range", {})
    vs_fr = fa.D_live / fa.firstrun_dead
    below_sept = fa[fa.D_live < fa.sept_dead]
    ev10 = f"""{R.figure(F['10a_final_tables']['F11'], 'Figure 10.1') if 'F11' in F.get('10a_final_tables', {}) else ''}
{R.table(pd.DataFrame(frows), ["exp", "line", "src", "W", "D", "C", "B", "old", "t1", "drift"],
         ["Exp", "Line", "Source", "90% window live–dead (run 2 primary)", "Rule D centre [95% CI]", "Rule C (sensitivity)",
          "Rule B live–dead", "First run · September (live/dead)", "Untreated called dead at T1: 90% window / D / C / B (%)",
          "Drift tier"])}
<p>† XDP10: interval from cells redrawn within 2 wells (too narrow). Transferred rows show the point value only; see
Step 8 for their resampling intervals. MRID-format threshold files (cell_line, live_threshold, dead_threshold; for the
single-cutoff rules live = dead): <code>10a_final_tables/thresholds/window90_primary/</code>, <code>window95_sensitivity/</code>,
<code>track1_ruleD/</code>,
<code>track1_ruleC_sensitivity/</code> and <code>track2_ruleB/</code>, one file per experiment.</p>"""
    out.append(step_block(
        "s10", "10", "Final thresholds, both tracks",
        "What cutoffs go into MRID, and how do they compare with the first run and September?",
        "<p>Direct cutoffs (Step 6) where usable, transferred cutoffs (Step 8, planned live-SD form) elsewhere, for track 1 "
        "(rule D, with rule C as its sensitivity) and track 2 (rule B).</p>", ev10,
        f"<p>Untreated cells called dead at T1 now range {100 * t1.get('W90', [np.nan, np.nan])[0]:.1f}–{100 * t1.get('W90', [np.nan, np.nan])[1]:.1f}% "
        f"with the 90% window (run 2 primary), {100 * t1.get('D', [np.nan, np.nan])[0]:.1f}–{100 * t1.get('D', [np.nan, np.nan])[1]:.1f}% "
        f"under the rule D single cutoff, {100 * t1.get('C', [np.nan, np.nan])[0]:.1f}–{100 * t1.get('C', [np.nan, np.nan])[1]:.1f}% under C and "
        f"{100 * t1.get('B', [np.nan, np.nan])[0]:.1f}–{100 * t1.get('B', [np.nan, np.nan])[1]:.1f}% under B - measured per line, "
        f"instead of 9.1% everywhere. Rule D cutoffs are {rng_txt(vs_fr.min(), vs_fr.max())}× the first run's hand-picked dead "
        f"cutoffs; they are above the September dead cutoffs in {len(fa) - len(below_sept)} of {len(fa)} cases and below in "
        f"{', '.join(below_sept.exp + ' ' + below_sept.line.map(SHORT))}. Next: MRID log-odds with each threshold set, to "
        "compare the tracks on odds ratios (D23).</p>",
        decision_refs(["D21", "D22", "D23"], decs)))
    return "\n".join(out)


SET_NAMES = {k: (lambda v: v[0].upper() + v[1:])(SETS_.LABELS.get(k, k))
             for k in ["W90", "W95", "D_sd", "D_abs", "C_sd", "C_abs", "B", "D_sd_drift", "W90_shared"]}
EXP_SHORT = lambda e: str(e).replace("-RGEDI", "")  # noqa: E731


def or_ci(v, lo, hi):
    if v is None or pd.isna(v):
        return "–"
    return f"{v:.2f} [{lo:.2f}, {hi:.2f}]"


def steps_11_to_13(decs, F, S) -> str:
    s11 = S.get("11_mrid_inputs", {})
    if not s11:
        return ""
    out = []
    LO = LOCAL_OUT / "12_logodds"
    # ---- Step 11 ----------------------------------------------------------------------------------------
    rows = [{"set": SET_NAMES[k], "cells": f"{v['cells_classified']:,}", "dead": f"{v['dead']:,}",
             "frac": f"{100 * v['death_frac']:.1f}%"} for k, v in s11["sets"].items()]
    ev11 = f"""{R.table(pd.DataFrame(rows), ["set", "cells", "dead", "frac"],
        ["Threshold set", "Cells classified", "Called dead", "Share called dead (all timepoints)"], num_cols=["cells", "dead", "frac"])}
<p>Cells used in every set: in-scope lines, untreated wells only (no S.A. wells, no siRNA wells), and for XDP1 (whole plate
dosed) all wells up to T10 - which includes XDP1's T7, flagged in Step 2 as a plate-wide jump in high ratios. Rule B drops
the cells between its two cutoffs, so it classifies fewer cells. The time-anchored set moves the cutoff with the untreated
live peak at each timepoint for the drifting cases ({', '.join(s11.get('drifting_cases', []))}). The live/dead calls reproduce
MRID's own <code>mrid_build_combined_livedead()</code> exactly (XDP5, rule B: identical live and dead counts in all 645 well ×
timepoint groups). Within an experiment, lines can have cutoffs from different sources (one fitted directly, another
transferred, e.g. XDP1 14C vs 09B/63D); Step 13 shows what that does.</p>"""
    out.append(step_block(
        "s11", "11", "Live/dead calls for MRID",
        "Which cells go into MRID, and how are they called live or dead under each threshold set?",
        "<p>One MRID input per threshold set (D23): rule D centre with a 90% and a 95% window (D28; run 2), rule D single cutoff with live-SD (run 1 primary) and absolute "
        "transfer (D21), rule C with both transfer forms, rule B (track 2), and rule D with time-anchored cutoffs for drifting cases "
        "(D22). Each cell is live below the live cutoff, dead above the dead cutoff, and dropped in between (MRID's rule).</p>",
        ev11, "<p>The sets differ only in the cutoffs; the cells and wells are the same.</p>",
        decision_refs(["D21", "D22", "D23"], decs)))

    # ---- Step 12 ----------------------------------------------------------------------------------------
    hours = R.step_csv(f"12_logodds/{P}", "time_index_hours.csv")
    kept = sorted(hours.Hour.astype(float).tolist()) if len(hours) else []
    log = (LO / P / "mrid_log.txt").read_text() if (LO / P / "mrid_log.txt").exists() else ""
    import re as _re
    dropped = sorted({int(h) for h in _re.findall(r"\|\s*(\d+)\|\s*1\|", log)})
    data = R.step_csv(f"12_logodds/{P}", "Data.csv")
    hr_exp = {}
    if len(data):
        dd = data[["Hour", "Experiment"]].drop_duplicates()
        for h, g in dd.groupby("Hour"):
            hr_exp[float(h)] = sorted(g.Experiment.map(EXP_SHORT))
    eph = R.step_csv(f"12_logodds/{P}", "experiments_per_hour.csv")
    eph_rows = []
    if len(eph):
        for h, g in eph.groupby("Hour"):
            r_ = {"h": f"{int(h)} h"}
            for l in figs.LINE_ORDER:
                ex = sorted(g[g.CellLine == l].Experiment.map(EXP_SHORT), key=lambda x: EXPS.index(x))
                r_[l] = ", ".join(ex) if ex else "–"
            eph_rows.append(r_)
    warn_rows, n_conv = [], 0
    for k in SET_NAMES:
        w1 = R.step_csv(f"12_logodds/{k}", "warnings.csv")
        w2 = R.step_csv(f"12_logodds/{k}", "warnings_12b.csv")
        c1 = int(w1.warning.str.contains("converge").sum()) if len(w1) else 0
        c2 = int(w2.warning.str.contains("converge").sum()) if len(w2) else 0
        n_conv += c1 + c2
        warn_rows.append({"set": SET_NAMES[k], "mrid": c1, "extra": c2})
    chk = R.step_json("13_or_figures", "checks")
    rd = {c["id"].split(".")[0]: c["value"] for c in chk if c["id"].endswith("mrid_rederived")}
    dev = max((v["max_abs_log_ratio"] for v in rd.values()), default=float("nan"))
    nre = rd.get(P, {}).get("n", 0)
    nre_sh = rd.get(f"{P}_shared", {}).get("n", 0)
    hr_sh = R.step_csv(f"12_logodds/{P}_shared", "time_index_hours.csv")
    ev12 = f"""<ul class="summary-list">
<li><b>Model (MRID, unchanged):</b> binomial mixed model of dead vs live counts per well and timepoint, fitted for each pair of
lines, with experiment and well-within-experiment random effects, line, timepoint (as categories, so the odds ratio can differ
at each timepoint) and their interaction. The odds ratio at a timepoint is the comparison line's odds of death divided by the
reference line's. The model assumes the line difference is the same in every experiment (Step 13 tests this).</li>
<li><b>Time axis:</b> MRID keeps only hours imaged in at least two experiments: {len(kept)} timepoints
({int(kept[0]) if kept else '–'}–{int(kept[-1]) if kept else '–'} h). Dropped: {', '.join(f'{h} h' for h in dropped) or 'none'}
(XDP1's 12-hour offsets and XDP11's last images). Which experiments contribute changes along the axis (table below), so a
change over time can also be a change in which experiments are present.</li>
<li><b>Checks:</b> every threshold set ran for all 12 ordered line pairs × {len(kept)} timepoints (the XDP5/10/11-only run:
× {len(hr_sh)} timepoints, {int(hr_sh.Hour.min()) if len(hr_sh) else '–'}–{int(hr_sh.Hour.max()) if len(hr_sh) else '–'} h). An
independent re-derivation from named coefficients matches MRID's odds ratios for all {nre} pair × timepoint values per set
({nre_sh} for the XDP5/10/11-only run; largest log-ratio difference {dev:.0e}).
{n_conv} model fits across all sets reported a convergence warning (table).</li>
<li><b>Fixed during this step:</b> loading <code>plyr</code> after <code>dplyr</code> silently disables MRID's "drop hours seen
in one experiment" filter; the driver now loads them in the safe order and asserts it.</li></ul>
<h3>Experiments contributing at each hour ({PL})</h3>
{R.table(pd.DataFrame(eph_rows), ["h"] + figs.LINE_ORDER, ["Hour"] + [SHORT[l] for l in figs.LINE_ORDER])}
<h3>Convergence warnings</h3>
{R.table(pd.DataFrame(warn_rows), ["set", "mrid", "extra"], ["Threshold set", "MRID per-timepoint fits", "Step 13 models"], num_cols=["mrid", "extra"])}"""
    out.append(step_block(
        "s12", "12", "MRID log-odds",
        "How does MRID turn the live/dead calls into odds ratios between lines?",
        "<p>MRID's own functions (<code>MRID_Logodd_Functions_June2026.R</code>), all experiments combined, nonlinear time, "
        "every line used as a reference.</p>", ev12,
        "<p>The odds ratios come from MRID as Julia knows it. Two properties matter for reading them: the time axis mixes "
        "experiments, and the model treats each line difference as identical in every experiment.</p>"))

    # ---- Step 13 ----------------------------------------------------------------------------------------
    pooled = R.step_csv("13_or_figures", "pooled_or_by_set.csv")
    f13 = F.get("13_or_figures", {})
    if not len(pooled):
        return "\n".join(out)
    main_sets = [k for k in SET_NAMES if not k.endswith("_shared")]
    pt = pooled[pooled.set.isin(main_sets)].copy()
    pt["o"] = pt.set.map({k: i for i, k in enumerate(main_sets)})
    pt = pt.sort_values(["Reference", "Compare", "o"])
    prow = [{"set": SET_NAMES[r.set], "pair": f"{SHORT[r.Compare]} vs {SHORT[r.Reference]}",
             "mrid": f"{r.mridpool_OR:.2f}", "lxe": or_ci(r.lxe_OR, r.lxe_lo, r.lxe_hi),
             "meta": or_ci(r.meta_all_OR, r.meta_all_lo, r.meta_all_hi) + (f" (I² {100 * r.meta_all_I2:.0f}%)" if pd.notna(r.meta_all_I2) else ""),
             "rep": f"{r.experiments_ci_above_1} / {r.experiments_ci_below_1} / {r.n_experiments}",
             "shared": or_ci(r.meta_shared_OR, r.meta_shared_lo, r.meta_shared_hi),
             "nox1": or_ci(r.meta_noXDP1_OR, r.meta_noXDP1_lo, r.meta_noXDP1_hi),
             "same": or_ci(r.meta_samesource_OR, r.meta_samesource_lo, r.meta_samesource_hi) + f" (k={int(r.meta_samesource_k)})"}
            for r in pt.itertuples()]
    figs_a = "".join(f'<details><summary>{SET_NAMES[k]}: odds ratios over time</summary><div class="inner">{R.figure(f13[f"Fa_{k}"])}</div></details>'
                     for k in main_sets if f"Fa_{k}" in f13 and k != P)
    figs_c = "".join(f'<details><summary>{SET_NAMES[k]}: per-experiment forest</summary><div class="inner">{R.figure(f13[f"Fc_{k}"])}</div></details>'
                     for k in main_sets if f"Fc_{k}" in f13 and k != P)
    xdp = pt[pt.Compare.str.startswith("XDP")]
    v14 = xdp[xdp.Reference == "CTR-33114.C"]
    v13 = xdp[xdp.Reference == "CTR-33113.2I"]
    n14_incl = int(((v14.lxe_lo <= 1) & (v14.lxe_hi >= 1)).sum())
    n14_mridsig = int((v14.mridpool_OR > 1).sum())
    i2 = pt.meta_all_I2.dropna()
    meta_excl = v14[(v14.meta_all_lo > 1) | (v14.meta_all_hi < 1)]
    meta_txt = ("" if not len(meta_excl) else
                f" (random-effects meta-analysis: {len(v14) - len(meta_excl)} of {len(v14)}; exception "
                + "; ".join(f"{SET_NAMES[r.set]}, {SHORT[r.Compare]} vs 14C {or_ci(r.meta_all_OR, r.meta_all_lo, r.meta_all_hi)}"
                            for r in meta_excl.itertuples()) + ")")
    pe1 = R.step_csv(f"12_logodds/{P}", "OR_time_averaged_per_experiment.csv").rename(columns=lambda c: c.split(".")[0])
    big = []
    for cmp in ("XDP-33363.D", "XDP-33109.2B"):
        g = pe1[(pe1.Reference == "CTR-33114.C") & (pe1.Compare == cmp)].sort_values("OR", ascending=False)
        if len(g) > 1:
            big.append(f"{SHORT[cmp]} vs 14C: largest in {EXP_SHORT(g.Experiment.iloc[0])} ({g.OR.iloc[0]:.2f}), next "
                       f"{EXP_SHORT(g.Experiment.iloc[1])} ({g.OR.iloc[1]:.2f})")
    srcmap = R.step_csv("10a_final_tables", "thresholds/thresholds_all.csv")
    _b13 = pt[(pt.Reference == "CTR-33113.2I") & (pt.Compare == "XDP-33109.2B")]
    excl13 = [r.set for r in _b13.itertuples() if r.lxe_hi < 1 or r.lxe_lo > 1]
    p1 = pt[pt.set == P].set_index(["Reference", "Compare"])

    def g(ref, cmp, col):
        try:
            return p1.loc[(ref, cmp), col]
        except KeyError:
            return np.nan
    ev13 = f"""<h3>13.1 Each line against each control line, over time ({PL})</h3>
{R.figure(f13[f'Fa_{P}'], 'Figure 13.1') if f'Fa_{P}' in f13 else ''}
<p>Each curve pools whichever experiments contain both lines at that hour (Step 12 table); 13I is only in XDP5, XDP10 and
XDP11, so its curves and the 14C curves do not rest on the same experiments.</p>{figs_a}
<h3>13.2 All line pairs, on the three experiments that contain all four lines ({PL})</h3>
{R.figure(f13[f'Fb_{P}_shared'], 'Figure 13.2') if f'Fb_{P}_shared' in f13 else ''}
{('<details><summary>Same, all experiments (each cell then rests on a different set of experiments)</summary><div class="inner">' + R.figure(f13[f'Fb_{P}']) + '</div></details>') if f'Fb_{P}' in f13 else ''}
<h3>13.3 Reproducibility across experiments</h3>
<p>Time-averaged odds ratio (one value per line pair) fitted per experiment with a well random effect. Pooled two ways:
with MRID's structure (outline diamond), and with an added line-by-experiment random effect so the pooled interval includes
how much the line difference varies between experiments (filled diamond). Open circles / * mark experiments where the two
lines' cutoffs come from different sources (one fitted directly, one transferred).</p>
{R.figure(f13[f'Fc_{P}'], 'Figure 13.3') if f'Fc_{P}' in f13 else ''}{figs_c}
<h3>13.4 All threshold sets: pooled estimates and sensitivity</h3>
{R.table(pd.DataFrame(prow), ["set", "pair", "mrid", "lxe", "meta", "rep", "shared", "nox1", "same"],
         ["Threshold set", "Pair", "Pooled, MRID structure", "Pooled, line × experiment [95% CI]",
          "Random-effects meta-analysis [95% CI] (I²)", "Experiments: CI above 1 / below 1 / total",
          "Meta, XDP5/10/11 only", "Meta, without XDP1", "Meta, same cutoff source only"])}
{R.figure(f13['Fd'], 'Figure 13.4') if 'Fd' in f13 else ''}"""
    take = (f"<p>Per experiment, the line differences vary far more than sampling allows (random-effects I² "
            f"{100 * i2.min():.0f}–{100 * i2.max():.0f}%). "
            f"With MRID's structure, which ignores that variation, both XDP lines show more death than 14C in every threshold set "
            f"({n14_mridsig} of {len(v14)} set × line estimates above 1). Once the line-by-experiment variation is allowed for, "
            f"the 95% interval includes 1 in {n14_incl} of {len(v14)} (line × experiment model){meta_txt} ({PL}: 09B vs 14C "
            f"{or_ci(g('CTR-33114.C', 'XDP-33109.2B', 'lxe_OR'), g('CTR-33114.C', 'XDP-33109.2B', 'lxe_lo'), g('CTR-33114.C', 'XDP-33109.2B', 'lxe_hi'))}, "
            f"63D vs 14C {or_ci(g('CTR-33114.C', 'XDP-33363.D', 'lxe_OR'), g('CTR-33114.C', 'XDP-33363.D', 'lxe_lo'), g('CTR-33114.C', 'XDP-33363.D', 'lxe_hi'))}). "
            f"Against 13I, 09B's point estimates are below 1 under rules D and C and above 1 under rule B, but with the "
            f"between-experiment variation included the interval excludes 1 only in "
            f"{', '.join(SET_NAMES[k] for k in excl13) or 'no set'}; 63D vs 13I is near 1 in every set. "
            f"The apparent difference between the two control lines as references is largely a difference in experiments: in "
            f"XDP5/10/11, the experiments that contain 13I, 09B vs 14C is {g('CTR-33114.C', 'XDP-33109.2B', 'meta_shared_OR'):.2f} "
            f"({PL}). The largest per-experiment odds ratios ({PL}: {'; '.join(big)}) come from XDP1, where 14C's cutoff "
            f"was fitted directly but 09B's and 63D's were transferred, and XDP3, where all cutoffs were transferred. Taken "
            f"together: the data do not show a reproducible difference in death between the XDP and control lines.</p>")
    out.append(step_block(
        "s13", "13", "Odds ratios across cell lines",
        "Do the XDP lines die more than the control lines, against each control line, under each threshold set, and is "
        "that reproducible across experiments? (Julia's request.)",
        "<p>Figures from the MRID odds ratios of Step 12, plus time-averaged odds ratios per experiment pooled with and without "
        "between-experiment variation, a random-effects meta-analysis, and sensitivity views (only the experiments that contain "
        "all four lines; without XDP1; only experiments where both lines' cutoffs have the same source).</p>",
        ev13, take, decision_refs(["D05", "D23", "D24", "D25"], decs),
        "<p>Odds ratios that changed direction between threshold sets, or that disagreed between experiments. Disagreement "
        "between experiments is the dominant feature; direction also changes between sets for some pairs (table).</p>"))
    return "\n".join(out)


def pfmt(p) -> str:
    if p is None or (isinstance(p, float) and pd.isna(p)):
        return "–"
    return "<0.0001" if p < 1e-4 else f"{p:.4f}"


def rng_txt(lo, hi, nd=2) -> str:
    return f"{lo:.{nd}f}" if abs(lo - hi) < 10 ** -nd else f"{lo:.{nd}f}–{hi:.{nd}f}"



CNT_STEPS = ["14_count_tables", "15_sa_capture", "16b_retention_fe", "17_count_synthesis"]
PAIRS17 = [("XDP-33109.2B", "CTR-33114.C"), ("XDP-33363.D", "CTR-33114.C"), ("CTR-33113.2I", "CTR-33114.C"),
           ("XDP-33109.2B", "CTR-33113.2I"), ("XDP-33363.D", "CTR-33113.2I")]


def pair_txt(c, r) -> str:
    return f"{SHORT[c]} vs {SHORT[r]}"


def wk(ratio_per_day, nd=2) -> str:
    """Relative live-cell loss per week (> 1: the first line loses live cells faster) from a daily retention ratio."""
    return f"{ratio_per_day ** -7:.{nd}f}"


def live_ratio_table() -> pd.DataFrame:
    """Live and detected cells at the last reference timepoint / at k0, per line x experiment (primary set)."""
    l = R.step_csv("14_count_tables", "counts_long.csv")
    k0 = R.step_csv("14_count_tables", "k0.csv").set_index("exp").k0
    l = l[(l.set == P) & l.reference & ~l.missing]
    rows = []
    for e in EXPS:
        g = l[(l.exp == e) & (l.tp >= k0[e])]
        last = g.tp.max()
        b, en = g[g.tp == k0[e]].set_index("well"), g[g.tp == last].set_index("well")
        ws = [w for w in b.index.intersection(en.index) if b.loc[w, "L"] >= 50]
        for line in figs.LINE_ORDER:
            w = [x for x in ws if b.loc[x, "line"] == line]
            if w:
                rows.append({"exp": e, "line": line, "k0": int(k0[e]), "last": int(last), "wells": len(w),
                             "L": en.loc[w, "L"].sum() / b.loc[w, "L"].sum(), "N": en.loc[w, "N"].sum() / b.loc[w, "N"].sum()})
    return pd.DataFrame(rows)


def steps_14_to_17(decs, F, S) -> str:
    s14, s15, s16, s17 = (S.get(k, {}) for k in CNT_STEPS)
    if not s17:
        return ""
    out = []
    PC = cfg.params()["counts"]
    cap = R.step_csv("15_sa_capture", "capture.csv")
    inv = R.step_csv("15_sa_capture", "capture_invariance.csv")
    prs = R.step_csv("15_sa_capture", "capture_pairs.csv")
    mod = R.step_csv("15_sa_capture", "gfp_moderator.csv")
    c1 = R.step_csv("16_retention", "c1_pairs.csv")
    c1f = R.step_csv("16_retention", "c1_fits.csv")
    dh = R.step_csv("16_retention", "c1_dharma.csv")
    c2 = R.step_csv("16b_retention_fe", "c2_per_experiment.csv")
    c2p = R.step_csv("16b_retention_fe", "c2_pooled.csv")
    j1 = R.step_csv("16b_retention_fe", "j1.csv")
    sim = R.step_csv("16b_retention_fe", "sim_summary.csv")
    po = R.step_csv("17_count_synthesis", "or_pooled_uncorrected_vs_corrected.csv")
    vc = R.step_csv("17_count_synthesis", "visible_vs_capture.csv")
    lap = R.step_json("16_retention", "laplace_check")
    prim = cap[cap.variant == "primary"]
    p1 = prim[prim.set == P]
    gate1 = p1[p1.valid.astype(bool)]
    # dropout itself is measured with the single cutoff (no buffer zone): with a window, killed cells that sit in the
    # zone are also "not called dead", so the window's capture fraction mixes dropout with the zone
    gsc = prim[(prim.set == "D_sd") & prim.valid.astype(bool)]
    drop_txt = (f"{rng_txt(gsc.c.min(), gsc.c.max())} of the S.A.-killed cells as dead (single cutoff; with the {PL} also "
                f"leaving the buffer-zone cells uncalled: {rng_txt(gate1.c.min(), gate1.c.max())})" if P != "D_sd"
                else f"{rng_txt(gate1.c.min(), gate1.c.max())} of the S.A.-killed cells as dead")
    dec = s17["decision"]
    ag = s17["agreement"]
    agr = ag[P]
    j1d = j1[j1.set == P]
    jpass = [e for e in EXPS if e in set(j1d[j1d.passes.astype(bool)].exp)]
    eqv = {r.exp: str(r.equal_visibility) for r in inv[(inv.basis == "valid") & (inv.set == P) & (inv.variant == "primary")].itertuples()}
    _xo = po[(po.set == P) & po.Compare.str.startswith("XDP") & (po.kind.isin(
        ["MRID (uncorrected)", "S.A.-corrected, D26 as frozen", "S.A.-corrected, D27 gated"]))]
    _ex = int(((_xo.lo > 1) | (_xo.hi < 1)).sum())
    or_txt = ("Every track-1 XDP-vs-control odds ratio has a 95% interval that includes 1, with or without the correction "
              "(Step 17)." if _ex == 0 else
              f"{_ex} track-1 XDP-vs-control odds-ratio intervals (across the uncorrected and corrected versions) exclude 1 "
              "(Step 17).")

    # ---- answer box --------------------------------------------------------------------------------------
    answer = f"""<div class="box takeaway"><span class="k">The short answer to "do we use cell counts, and what is the sound way?"</span>
<p><b>Before this step, no.</b> MRID compares the share of dead cells among cells that are still detected; a dying cell
that has lost its GFP is simply missing from both the numerator and the denominator.</p>
<p><b>The sound way</b> is to model the two readouts together. For each line: (1) the number of live cells falls with
death and also changes for other reasons - cells becoming visible as GFP matures, detection changes, merging:
<i>d log(live)/dt = −death rate + other gains and losses</i>; (2) the visible dead share is roughly <i>death rate × how
long a dead cell stays visible</i>, and GFP dropout shortens the second factor. Three unknowns per line, two readouts: a
death comparison between lines needs one of the other two pieces from outside. We tried both routes - S.A. wells to
measure how much death stays visible (Step 15), and a joint model asking whether visible death explains the loss of live
cells (Step 16) - and checked each method on simulated plates with known truth.</p>
<p><b>Result.</b> Dropout is large: at the S.A. readout GEDI still sees only {drop_txt} (primary readout, cases passing the validity check). But neither route fixes the line
comparison reliably: the S.A. correction changes with the readout timepoint, and counts and odds ratios point the same way
in only {agr['sign_agree']} of {agr['n']} line pairs × experiments - no better than chance. The calibration test does not
reject in {' and '.join(jpass) or 'no experiment'} only; in XDP11 the S.A. wells show unequal capture between lines, which
that test assumes away, and XDP1 has no untreated wells to check it. {or_txt}</p>
<p><b>What to do now.</b> Keep the MRID odds ratios (with the line × experiment term, D25) as the death comparison and show
the S.A.-corrected range and the live-cell retention next to them as sensitivity analyses (D27, pending). <b>What would
make a correction possible:</b> imaging S.A. wells at several short intervals after dosing (to measure each line's kill
timing and how long its dead cells stay visible), or segmenting on RFP or both channels so that dying cells stay
detectable.</p></div>"""

    # ---- Step 14 ----------------------------------------------------------------------------------------
    k0 = R.step_csv("14_count_tables", "k0.csv")
    lr = live_ratio_table()
    k0_rows = [{"exp": r.exp, "k0": f"T{r.k0}", "half": r.first_half,
                "t1": f"{r.mean_N_T1:.0f}", "atk0": f"{getattr(r, f'mean_N_T{r.k0}'):.0f}",
                "zero": s14["qc"][r.exp]["zero"], "dip": s14["qc"][r.exp]["dip"],
                "low": s14["low_baseline_wells"][P][r.exp]} for r in k0.itertuples()]
    piv = lr.assign(v=lambda x: x.L.map(lambda v: f"{v:.2f}")).pivot(index="exp", columns="line", values="v").reindex(EXPS)
    piv = piv.reindex(columns=[l for l in figs.LINE_ORDER if l in piv.columns]).reset_index()
    spread = lr.groupby("exp").L.agg(["min", "max"])
    big = spread.assign(d=spread["max"] - spread["min"]).sort_values("d", ascending=False)
    ck14 = {c["id"]: c for c in R.step_json("14_count_tables", "checks")}
    ok11 = all(v["status"] == "PASS" for k, v in ck14.items() if k.endswith("matches_step11"))
    ok3 = ck14.get("N_matches_step3", {}).get("status") == "PASS"
    x10 = lr[(lr.exp == "XDP10") & (lr.line == "CTR-33113.2I")]
    ev14 = f"""{answer}
<h3>14.1 Start of the counting window (k0) and quality control</h3>
{R.table(pd.DataFrame(k0_rows), ["exp", "k0", "half", "t1", "atk0", "zero", "dip", "low"],
         ["Exp", "k0", "Searched", "Cells per reference well, T1", "at k0", "Well × timepoints with no cells",
          "Dips (missing)", "Wells < 50 live cells at k0"], num_cols=["t1", "atk0", "zero", "dip", "low"])}
<p>Reference wells = untreated wells; XDP1 (whole plate dosed) uses all its wells up to T10, before dosing. k0 is the
timepoint in the first half of the run where the mean number of detected cells per reference well (all lines together)
peaks, so the window starts after cells have stopped becoming newly visible as GFP matures; it is chosen without looking at
lines or cutoffs. A well × timepoint is treated as missing when it has no cells, or when its count falls below half of both
neighbours and then recovers (cells cannot come back, so such a dip is an imaging failure; XDP8's T3 has many). Checks:
live and dead counts in reference wells equal the Step 11 MRID inputs in every line × timepoint for every threshold set
({'pass' if ok11 else 'FAIL'}); detected cells equal Step 3's counts in every well × timepoint ({'pass' if ok3 else 'FAIL'}).</p>
<h3>14.2 Live cells at the end of the window relative to k0 (reference wells, {PL})</h3>
{R.table(piv, ["exp"] + [c for c in piv.columns if c != "exp"], ["Exp"] + [SHORT[c] for c in piv.columns if c != "exp"],
         num_cols=[c for c in piv.columns if c != "exp"])}
{R.figure(F['14_count_tables']['F14_1'], 'Figure 14.1')}"""
    take14 = (f"<p>Live-cell counts move differently in different lines of the same plate: the widest spread is in "
              f"{big.index[0]} ({big.iloc[0]['min']:.2f}–{big.iloc[0]['max']:.2f} of the k0 count) and {big.index[1]} "
              f"({big.iloc[1]['min']:.2f}–{big.iloc[1]['max']:.2f})"
              + (f"; XDP10 13I even ends at {x10.L.iloc[0]:.2f} of its k0 count" if len(x10) and x10.L.iloc[0] > 1 else "")
              + ". Some of this is death and some is cells becoming visible or invisible for other reasons; a count alone "
              "cannot tell which.</p>")
    out.append(step_block(
        "s14", "14", "Cell counts over time",
        "Do we use the number of cells over time anywhere in the death estimate, and how many live, dead and detected cells "
        "does each well have at each timepoint?",
        "<p>For every well × timepoint: live (L), dead (D) and all detected cells (N), with the same live/dead call as "
        "Step 11 for every threshold set, now including the S.A. wells. The quality-control rules and the start of the "
        "counting window were fixed before the formal fits, after pilot numbers had been seen (D26).</p>",
        ev14, take14, decision_refs(["D26"], decs)))

    # ---- Step 15 ----------------------------------------------------------------------------------------
    def cell(r):
        flag = "" if r.valid else (" (fails gate)" if r.usable else " (not usable)")
        return f"{r.c:.2f} [{r.c_lo:.2f}, {r.c_hi:.2f}]{flag}"
    rows = []
    for exp in PC["capture_tps"]:
        for line in figs.LINE_ORDER:
            a = prim[(prim.exp == exp) & (prim.line == line) & (prim.set == P)]
            b = prim[(prim.exp == exp) & (prim.line == line) & (prim.set == "D_sd")]
            if not len(a):
                continue
            a = a.iloc[0]
            rows.append({"exp": exp, "line": SHORT[line] + (FEW if a.few_wells else ""), "tp": f"T{a.t0}→T{a.t}",
                         "killed": f"{a.killed_frac:.2f} [{a.killed_lo:.2f}, {a.killed_hi:.2f}]",
                         "det": f"{a.det_ratio:.2f} [{a.det_lo:.2f}, {a.det_hi:.2f}]",
                         "c": cell(a), "cB": cell(b.iloc[0]) if len(b) else "–"})

    def vis_cell(exp, set_, basis):
        iv = inv[(inv.exp == exp) & (inv.set == set_) & (inv.variant == "primary") & (inv.basis == basis)]
        if not len(iv) or pd.isna(iv.iloc[0].equal_visibility):
            return R.badge("muted", "no valid readout"), "–"
        iv = iv.iloc[0]
        bad = prs[(prs.exp == exp) & (prs.set == set_) & (prs.variant == "primary") & (prs.basis == basis)
                  & ~prs.ci_includes_1.astype(bool)]
        badtxt = "; ".join(f"{SHORT[x.line_A]}/{SHORT[x.line_B]} {x.ratio:.2f} [{x.lo:.2f}, {x.hi:.2f}]" for x in bad.itertuples())
        eq = str(iv.equal_visibility) == "True"
        return (R.badge("PASS" if eq else "WARN", ("no difference detected" if eq else "differs") + f" · Q p = {pfmt(iv.Q_p)}")
                + (f" ({iv.lines_excluded} excluded)" if isinstance(iv.lines_excluded, str) and iv.lines_excluded else ""),
                badtxt or "none")
    irows = []
    for exp in PC["capture_tps"]:
        r = {"exp": exp}
        r["v26"], r["p26"] = vis_cell(exp, P, "usable")
        r["v27"], r["p27"] = vis_cell(exp, P, "valid")
        r["vB"], _ = vis_cell(exp, "D_sd", "valid")
        irows.append(r)
    t9 = cap[(cap.set == P) & (cap.variant == "t9")]
    t10 = p1[p1.exp == "XDP10"]
    pre11 = cap[(cap.set == P) & (cap.variant == "pre11") & cap.valid.astype(bool)]
    eq11 = inv[(inv.basis == "valid") & (inv.set == P) & (inv.variant == "pre11")]
    plc = cap[(cap.variant == "placebo") & (cap.set == P)]
    plc_out = plc[(plc.killed_lo > 0) | (plc.killed_hi < 0)]
    m1 = mod[mod.set == P].iloc[0] if len(mod[mod.set == P]) else None
    x5 = p1[p1.exp == "XDP5"]
    t9g = t9.set_index("line")
    t9txt = "; ".join(f"{SHORT[l]} {t9g.loc[l].c:.2f} [{t9g.loc[l].c_lo:.2f}, {t9g.loc[l].c_hi:.2f}]"
                      + ("" if bool(t9g.loc[l].valid) else ", fails the gate") for l in figs.LINE_ORDER if l in t9g.index)
    t10g = t10.set_index("line")
    t9share = "; ".join(f"{SHORT[l]} {t9g.loc[l].killed_frac / t10g.loc[l].killed_frac:.2f}"
                        for l in figs.LINE_ORDER if l in t9g.index and l in t10g.index)
    p6 = cap[(cap.set == P) & (cap.variant == "pre6") & cap.valid.astype(bool)]
    x5_pre6_txt = ("; with the alternative baseline T6 (T7 is a plate-wide artefact day) "
                   + ", ".join(f"{SHORT[r.line]} c {r.c:.2f} [{r.c_lo:.2f}, {r.c_hi:.2f}]" for r in p6.itertuples())
                   + " passes, the other lines still fail") if len(p6) else ""
    ia = inv[(inv.basis == "valid") & (inv.set == "D_abs") & (inv.variant == "primary") & (inv.exp == "XDP3")]
    pa = prs[(prs.basis == "valid") & (prs.set == "D_abs") & (prs.variant == "primary") & (prs.exp == "XDP3")
             & ~prs.ci_includes_1.astype(bool)]
    d_abs_txt = (f"XDP3's cutoffs are all transferred; with absolute transfer offsets (the D21 alternative) its capture "
                 f"fractions differ between lines ({'; '.join(f'{SHORT[x.line_A]}/{SHORT[x.line_B]} {x.ratio:.2f} [{x.lo:.2f}, {x.hi:.2f}]' for x in pa.itertuples())}) - "
                 "another sign that c is fragile.") if len(ia) and str(ia.iloc[0].equal_visibility) == "False" else ""
    p11 = prs[(prs.basis == "valid") & (prs.set == P) & (prs.variant == "primary") & (prs.exp == "XDP11")
              & ~prs.ci_includes_1.astype(bool)]
    x11_txt = "; ".join(f"c ratio {SHORT[x.line_A]}/{SHORT[x.line_B]} {x.ratio:.2f} [{x.lo:.2f}, {x.hi:.2f}]" for x in p11.itertuples())
    s4 = sim[sim.scenario.str.startswith("S4 ")].iloc[0]
    s5 = sim[sim.scenario.str.startswith("S5")].iloc[0]
    ev15 = f"""<div class="box"><span class="k">The idea</span>
<p>Within one line on one plate, S.A. and untreated wells were plated, matured and imaged together. So the untreated wells
give the number of live and dead cells the S.A. wells would have had without S.A.; the difference is what S.A. did. Of the
live cells S.A. removed (K), the share still seen as GEDI-dead (E) is the <b>capture fraction c = E / K</b>. c below 1 means
the rest of the S.A.-killed cells have already lost GFP and dropped out.</p>
<pre class="mono">expected live, no dose    Lhat = L_SA(pre) × L_UT(post) / L_UT(pre)
live cells removed           K = Lhat − L_SA(post)
expected dead, no dose    Dhat = Lhat × odds_SA(pre) × odds_UT(post) / odds_UT(pre)     (odds = dead / live)
excess visible dead          E = D_SA(post) − Dhat
capture fraction             c = E / K</pre>
<p><b>Why it could matter for the odds ratios.</b> The dead share MRID models is roughly (death rate) × (how long a dead
cell stays visible). GFP dropout shortens the second factor. That lowers measured death in every line, and it changes an
odds ratio between two lines only if the lines differ in how long their dead cells stay visible. c is a proxy for that:
if two lines' dead cells are seen at different rates, c differs in proportion. It is an imperfect proxy - in the
simulation (16.3), dead cells that stay visible 1.5× longer give a capture ratio of only {s4.c_ratio_true_09B:.2f}, and a
line whose S.A. kill simply happens earlier after dosing gets a capture ratio of {s5.c_ratio_true_09B:.2f} with no difference
in visibility at all. S.A.-killed cells may also vanish faster than cells that die slowly on their own.</p>
<p><b>Validity check (D27, proposed after review).</b> Under a single cutoff, c above 1 is only possible if the
S.A. wells kept more detected cells than the no-dose expectation - then c is not a capture fraction but a sign that the
untreated wells are not a good counterfactual for the detected cells. A capture fraction is therefore used only if the
S.A. wells also lost detected cells (95% upper bound of "detected after dose ÷ expected" below 1).</p></div>
<h3>15.1 Capture fraction per line × experiment</h3>
{R.table(pd.DataFrame(rows), ["exp", "line", "tp", "killed", "det", "c", "cB"],
         ["Exp", "Line", "Pre → readout", "Live cells removed by S.A. (share) [95% CI]",
          "Detected cells after dose ÷ expected [95% CI]", f"Capture c, {PL} [95% CI]", "Capture c, single rule D cutoff"],
         num_cols=["killed"])}
<p>Readout = the last post-dose timepoint used for the thresholds: one image after the dose in XDP3 (inferred dose time)
and XDP11, two images after it in XDP5 and XDP10. {FEW.strip()} XDP10 has 2 S.A. wells per line, all in the edge column 22;
its intervals redraw the counts of those wells (Poisson) instead of redrawing wells, so they understate uncertainty.
Usable (D26): at least 10% of live cells removed, lower bound above 0. Passing the gate (D27): also detected cells fell.
Well bootstrap, B = {PC['B_capture']}.</p>
{R.figure(F['15_sa_capture']['F15_1'], 'Figure 15.1')}
<p><b>XDP5.</b> Its capture fractions are {rng_txt(x5.c.min(), x5.c.max())}: the S.A. wells kept as many or more detected cells
than expected ({rng_txt(x5.det_ratio.min(), x5.det_ratio.max())} of the expectation) while {rng_txt(x5.killed_frac.min(), x5.killed_frac.max())}
of live cells were removed over two imaging intervals; in the placebo comparison (no dose, about a week) the "removed" share
was {abs(plc.killed_frac).median():.2f} in the median case and up to {abs(plc.killed_frac).max():.2f}. None of the primary XDP5
readouts passes the gate{x5_pre6_txt}. These are not capture fractions.</p>
<h3>15.2 Is visibility the same in every line of an experiment?</h3>
{R.table(pd.DataFrame(irows), ["exp", "v26", "p26", "v27", "p27", "vB"],
         ["Exp", f"{PL}, D26 as frozen", "Line pairs that differ (ratio of c)", f"{PL}, with the gate (D27)",
          "Line pairs that differ", "Single rule D cutoff, with the gate"], raw_cols=["v26", "v27", "vB"])}
<p>Equal (D26): every pairwise 95% interval for the ratio of capture fractions includes 1, and the heterogeneity test
across lines has p ≥ {PC['invariance_alpha']}. With up to six pairwise tests plus the overall test, this rule calls a
difference fairly often even when there is none, and "no difference detected" mostly means the intervals are wide.
<b>Readout matters:</b> read one image after the dose (T9) instead of two (T10), XDP10's capture fractions differ between
lines ({t9txt}); at T10 they are {rng_txt(t10.c.min(), t10.c.max())} with no difference detected. Why is not known: the lines'
S.A. kill proceeds at different speeds (share of the T10 kill already present at T9: {t9share}), and the T9 values have wide
intervals. XDP11 with baseline T11 instead of T12: c {rng_txt(pre11.c.min(), pre11.c.max()) if len(pre11) else '–'},
{'no difference detected' if len(eq11) and str(eq11.iloc[0].equal_visibility) == 'True' else 'differences between lines'}.
{d_abs_txt} Placebo (T1 to the
pre-dose timepoint, before any dose): the "removed" share has a 95% interval including 0 in {len(plc) - len(plc_out)} of
{len(plc)} cases ({', '.join(f'{r.exp} {SHORT[r.line]} {r.killed_frac:+.2f}' for r in plc_out.itertuples()) or 'none outside'}),
so S.A.-designated and untreated wells can drift apart by this much over a week without any dose. Under track 2, c also
counts S.A.-killed cells left in rule B's dropped middle band, so it is not a visibility measure there.</p>
{R.figure(F['15_sa_capture']['F15_2'], 'Figure 15.2') if 'F15_2' in F['15_sa_capture'] else ''}
{f"<p>There is no clear link between a line's GFP brightness and its capture fraction (within-experiment slope of log c on log GFP {m1.slope_log_c_per_log_gfp:.2f} [{m1.lo:.2f}, {m1.hi:.2f}], {int(m1.n)} cases passing the gate); the interval is wide, so a moderate effect is not excluded.</p>" if m1 is not None else ""}"""
    late = gsc[gsc.exp.isin(["XDP3", "XDP10", "XDP11"])]
    take15 = (f"<p><b>Dropout is large.</b> At the S.A. readout, GEDI sees as dead only {rng_txt(late.c.min(), late.c.max())} "
              "of the live cells S.A. removed (single cutoff, cases passing the gate); the rest have already left detection. "
              "<b>Whether it differs between lines is not settled by these data.</b> With the gate, no difference between "
              f"lines is detected in {', '.join(e for e in EXPS if eqv.get(e) == 'True') or 'no experiment'} (wide intervals) "
              f"and capture differs in {', '.join(e for e in EXPS if eqv.get(e) == 'False') or 'none'}"
              + (f" ({x11_txt})" if eqv.get("XDP11") == "False" and x11_txt else "")
              + (f"; {', '.join(e for e in EXPS if eqv.get(e) == 'nan')} "
                 f"{'has' if sum(eqv.get(e) == 'nan' for e in EXPS) == 1 else 'have'} no valid readout"
                 if any(eqv.get(e) == 'nan' for e in EXPS) else "")
              + ". But c changes with the readout timepoint (XDP10 T9 vs T10), possibly because the lines die at different "
              "speeds after dosing, so it is a weak basis for correcting odds ratios. Under the frozen D26 rule the "
              "corrected odds ratios would become primary; D27 proposes keeping the MRID odds ratios primary and showing the "
              "corrected ones as a sensitivity range (Step 17).</p>")
    out.append(step_block(
        "s15", "15", "How much S.A. death does GEDI see?",
        "Of the live cells S.A. removed, what share is still visible as GEDI-dead, and is that share the same in every line?",
        "<p>S.A. wells used as a calibration experiment for dropout (D26), at the S.A. readout timepoint; alternative "
        "pre-dose timepoints and XDP10's T9 as sensitivity; a validity check added after review (D27).</p>", ev15, take15,
        decision_refs(["D26", "D27"], decs),
        "<p>Capture fractions that differ between lines - then dropout biases the MRID odds ratios by the ratio of the "
        "two lines' visibility. It may (XDP11), but the capture fraction also moves with the readout, so the size of any bias "
        "cannot be pinned down from these data.</p>"))

    # ---- Step 16 ----------------------------------------------------------------------------------------
    def g1(df, fit, c, r):
        x = df[(df.fit == fit) & (df.Compare == c) & (df.Reference == r)]
        return x.iloc[0] if len(x) else None
    rows = []
    for c, r in PAIRS17:
        a, cc = g1(c1, P, c, r), g1(c2p, P, c, r)
        sens = []
        for fit in ("D_sd", "B", "N", f"{P}_T1", f"{P}_k0p1", f"{P}_noXDP10", f"{P}_no_hetero"):
            x = g1(c1, fit, c, r)
            sens.append(f"{wk(x.ratio_per_day)}{'*' if x.p < 0.05 else ''}" if x is not None else "–")
        rows.append({"pair": pair_txt(c, r),
                     "c1": f"{wk(a.ratio_per_day)} [{wk(a.upper)}, {wk(a.lower)}]" if a is not None else "–",
                     "p": pfmt(a.p) if a is not None else "–",
                     "sens": " · ".join(sens),
                     "c2": (f"{wk(cc.ratio_per_day)} [{wk(cc.upper)}, {wk(cc.lower)}]; new experiment "
                            f"[{wk(cc.pred_upper)}, {wk(cc.pred_lower)}]") if cc is not None and pd.notna(cc.pred_upper)
                     else (f"{wk(cc.ratio_per_day)} [{wk(cc.upper)}, {wk(cc.lower)}]" if cc is not None else "–")})
    jrows = []
    for r in j1.itertuples():
        if pd.notna(r.V_lo):
            vt = f"{r.V_hours:.0f} h [{r.V_lo:.0f}, {r.V_hi:.0f}]"
        elif r.b_lo_boot > 0:
            vt = "lines with more visible death lose fewer live cells"
        else:
            vt = "no clear relation"
        jrows.append({"exp": r.exp, "set": SHORT_SET.get(r.set, r.set), "V": vt,
                      "n": f"{int(r.n_lines)} · {r.collinearity_r2:.2f}", "wald": pfmt(r.wald_p),
                      "pass": R.badge("PASS" if r.passes else "WARN",
                                      r.status.replace("no relation", "opposite relation") if r.b_lo_boot > 0 else r.status)})
    srows = []
    for r in sim.itertuples():
        srows.append({"sc": r.scenario.split(" ", 1)[1], "or": f"{np.exp(r.OR_bias_09B):.2f}",
                      "orc": f"{np.exp(r.ORcorr_bias_09B):.2f}", "c2": f"{100 * r.c2_bias_09B:+.2f}",
                      "cov": f"{100 * min(r.c2_coverage_09B, r.c2_coverage_13I, r.c2_coverage_63D):.0f}%",
                      "rej": f"{100 * r.j1_reject_rate:.0f}%",
                      "V": (f"{r.j1_V_median:.0f} h (true {r.V_true_14C:.0f}"
                            + (f"; 09B {r.V_true_09B:.0f})" if r.V_true_09B != r.V_true_14C else ")"))
                      if pd.notna(r.j1_V_median) and r.j1_reject_rate < 0.5 else "test rejects"})
    d0 = dh.iloc[0] if len(dh) else None
    let_sd = float(c1f[c1f.fit == P].sd_LET.iloc[0]) if "sd_LET" in c1f and len(c1f) else float("nan")
    phi_hi = c2[(c2.fit == P) & [(c_, r_) in PAIRS17 for c_, r_ in zip(c2.Compare, c2.Reference)]].sort_values(
        "phi", ascending=False).head(3)
    s1 = sim[sim.scenario.str.startswith("S1")].iloc[0]
    s3 = sim[sim.scenario.str.startswith("S3")].iloc[0]
    s4b = sim[sim.scenario.str.startswith("S4b")].iloc[0]
    ev16 = f"""<div class="box"><span class="k">What a count model can and cannot say</span>
<p>The number of live cells a line keeps changes for two reasons: cells die (they leave the live pool whether or not they
stay visible as dead, so dropout does not matter here), and cells appear or vanish for other reasons (GFP maturing,
detection, merging). Comparing two lines on the same plate removes everything the lines share - imaging days, plate-wide
detection - but not reasons that differ between lines. So a difference in live-cell loss is a difference in death
<b>only if</b> non-death changes are the same in both lines, which Step 14 shows is not always true.</p></div>
<h3>16.1 Live-cell loss: each line vs a control line</h3>
<p>Values are the <b>control-to-line retention ratio per week</b>: the share of live cells the control keeps over a week
divided by the share the first line keeps. Above 1 = the first line keeps fewer of its live cells; 1.10 means it keeps
1/1.10 ≈ 91% as many as the control would (this is not a ratio of cells lost). Mixed model over all experiments: experiment ×
timepoint effects, line × experiment levels, a slope in days per line, line × experiment variation in that slope (the D25
term), shocks shared by all wells of a line at one timepoint (SD {let_sd:.3f} on the log scale), well level and slope, and
an observation-level term for extra noise. Its intervals use normal quantiles and its residual checks fail (below), so
they are probably too narrow.</p>
{R.table(pd.DataFrame(rows), ["pair", "c1", "p", "sens", "c2"],
         ["Pair", f"Mixed model, {PL} [95% CI]", "p",
          "Sensitivity: single rule D cutoff · rule B · all detected cells · counting window from T1 · from k0+1 · without XDP10 · without the line × experiment term (* p < 0.05)",
          "Per experiment, pooled (Hartung-Knapp) [95% CI]; range for a new experiment"])}
<p>Per-experiment version: Poisson model with well and timepoint effects and the line × timepoint contrasts; each pair's
slope is a straight line through its timepoint contrasts, with the standard error inflated when the contrasts scatter
around that line more than chance allows, with t quantiles (largest inflation: {', '.join(f'{r.exp} {SHORT[r.Compare]} vs {SHORT[r.Reference]} ×{np.sqrt(max(r.phi, 1)):.1f}' for r in phi_hi.itertuples())}).
Pooled over experiments with a random-effects model and small-sample (Hartung-Knapp) intervals; "range for a new
experiment" is the prediction interval. Diagnostics of the mixed model (simulated residuals): dispersion ratio
{d0.dispersion_ratio:.2f}, uniformity test p = {E(pfmt(d0.ks_p))} (the residual distribution is not what the model assumes),
median correlation of consecutive residuals within a well {d0.median_within_well_lag1:.2f}.{f" The fast fitting mode was checked against the slow Laplace fit on the pre-review version of the model: daily retention ratios agree to {lap['max_abs_log_ratio_per_day_diff']:.0e} on the log scale." if lap else ""}</p>
{R.figure(F['16b_retention_fe']['F16_1'], 'Figure 16.1')}
<h3>16.2 Does visible death explain live-cell loss? (calibration test)</h3>
<p>If visible GEDI death is what removes live cells, a line's live count should fall by its cumulative dead/live odds
divided by V, the hours a dead cell stays visible, and no other difference between lines should remain. Per experiment:
one coefficient for cumulative odds (well bootstrap, B = {PC['B_j1']}) and a robust test for remaining line differences.
"Not rejected" (D26: coefficient clearly negative and test p ≥ {PC['j1_alpha']}) means only that the data do not contradict
it: with 3-4 lines per experiment the cumulative odds are often almost a straight-line function of the line slopes
(R² {rng_txt(j1.collinearity_r2.min(), j1.collinearity_r2.max())}; column "lines · collinearity"), and on simulated 4-line
plates the test catches a 1.5× visibility difference only {100 * s3.j1_reject_rate:.0f}% of the time (likely less with 3
lines). It also assumes one V for all lines. The last column gives the reason when the test is not passed: line
differences remain (the robust test rejects), or the cumulative odds show no relation to live-cell loss.</p>
{R.table(pd.DataFrame(jrows), ["exp", "set", "V", "n", "wald", "pass"],
         ["Exp", "Cutoffs", "V, hours a dead cell stays visible [95% CI]", "Lines · collinearity (R²)",
          "Remaining line differences, p", "Calibration"], raw_cols=["pass"])}
{R.figure(F['16b_retention_fe']['F16_2'], 'Figure 16.2')}
<h3>16.3 Simulation: what each estimate recovers</h3>
<p>Simulated plates (4 lines, 24 untreated and 6 S.A. wells each, 10 daily images, imaging-day changes, well-to-well noise)
with known death rates (09B twice 14C), visibility and non-death changes; {int(sim.n.iloc[0])} runs per scenario. The same
estimators as on the real data.</p>
{R.table(pd.DataFrame(srows), ["sc", "or", "orc", "c2", "cov", "rej", "V"],
         ["Scenario", "Odds ratio ÷ true death-rate ratio (09B vs 14C)", "S.A.-corrected ÷ truth",
          "Retention slope − truth (% per day)", "Retention 95% CI coverage (lowest pair)", "Calibration test rejects",
          "V estimated (median)"], num_cols=["or", "orc", "c2", "cov", "rej"])}
{R.figure(F['16b_retention_fe']['F16_3'], 'Figure 16.3')}"""
    t1 = {pair_txt(c, r): g1(c1, P, c, r) for c, r in PAIRS17}
    sig = [k for k, v in t1.items() if v is not None and v.p < 0.05]
    noX = {pair_txt(c, r): g1(c1, f"{P}_noXDP10", c, r) for c, r in PAIRS17}
    T1 = {pair_txt(c, r): g1(c1, f"{P}_T1", c, r) for c, r in PAIRS17}
    t1only = [k for k, v in T1.items() if v is not None and v.p < 0.05 and k not in sig]
    lost = [k for k in sig if noX.get(k) is not None and noX[k].p >= 0.05]
    jremain = [e for e in EXPS if e in set(j1d[j1d.status.str.contains("line differences remain")].exp)]
    jnorel = [e for e in EXPS if e in set(j1d[j1d.status == "no relation to visible death"].exp)]
    take16 = (f"<p><b>Live-cell loss.</b> In the mixed model, {', '.join(sig) if sig else 'no pair'} clearly "
              f"differ{'s' if len(sig) == 1 else ''} from 1 ("
              + "; ".join(f"{k} {wk(v.ratio_per_day)} [{wk(v.upper)}, {wk(v.lower)}]" for k, v in t1.items() if k in sig)
              + " per week)"
              + (f"; without XDP10 {', '.join(lost)} no longer does" if lost else "")
              + (f"; with the window starting at T1, {', '.join(t1only)} also differ{'s' if len(t1only) == 1 else ''}" if t1only else "")
              + ". The per-experiment fits pooled with small-sample intervals give "
              + "; ".join(f"{pair_txt(c, r)} {wk(x.ratio_per_day)} [{wk(x.upper)}, {wk(x.lower)}]"
                          for (c, r) in PAIRS17 for x in c2p[(c2p.fit == P) & (c2p.Compare == c) & (c2p.Reference == r)].itertuples())
              + ". These are differences in net live-cell retention; they are death differences only where non-death "
              f"changes are equal. <b>Calibration ({PL}).</b> Not rejected in {', '.join(jpass) or 'no experiment'}"
              + (f"; line differences in live-cell loss beyond what GEDI death predicts in {', '.join(jremain)}" if jremain else "")
              + (f"; no relation between cumulative odds and live-cell loss in {', '.join(jnorel)} (uninformative)" if jnorel else "")
              + ". "
              f"<b>Simulation.</b> The odds ratio is unbiased when visibility is equal (S1 {np.exp(s1.OR_bias_09B):.2f}) and biased "
              f"by the visibility difference otherwise; the S.A. correction removes a difference in how often dead cells are "
              f"seen (S3 {np.exp(s3.ORcorr_bias_09B):.2f}), only partly fixes one in how long they stay visible, over-corrects it at "
              f"a two-image readout (S4b {np.exp(s4b.ORcorr_bias_09B):.2f}), and creates a bias when lines merely die at different "
              f"speeds after dosing (S5 {np.exp(s5.ORcorr_bias_09B):.2f}). The retention slope is unbiased only when non-death "
              "changes are equal.</p>")
    out.append(step_block(
        "s16", "16", "Live-cell retention",
        "Do the lines differ in how fast they lose live cells, and does visible GEDI death explain that loss?",
        "<p>Mixed model over all experiments (R, lme4), per-experiment models pooled with small-sample intervals, a "
        "calibration test joining counts and GEDI in one model, and a simulation with known truth.</p>",
        ev16, take16, decision_refs(["D25", "D26"], decs),
        "<p>A live-cell difference opposite to the GEDI odds ratio - it is common (Step 17), so neither readout can be taken "
        "as the death difference on its own.</p>"))

    # ---- Step 17 ----------------------------------------------------------------------------------------
    KINDS = [("MRID (uncorrected)", "MRID"), ("S.A.-corrected, D26 as frozen", "corrected, D26 as frozen"),
             ("S.A.-corrected, D27 gated", "corrected, gated (D27)")]
    SENS = ["S.A.-corrected, D27 gated, XDP10 read at T9", "S.A.-corrected, D27 gated, XDP11 baseline T11"]

    def orcell(set_, c, r, kind):
        x = po[(po.set == set_) & (po.Compare == c) & (po.Reference == r) & (po.kind == kind)]
        if not len(x):
            return "–"
        x = x.iloc[0]
        return f"{x.OR:.2f} [{x.lo:.2f}, {x.hi:.2f}] · {int(x.exp_ci_above_1)}↑ {int(x.exp_ci_below_1)}↓ of {int(x.k)}"
    rows, rowsB = [], []
    for c, r in PAIRS17:
        row = {"pair": pair_txt(c, r)}
        for kind, k in KINDS:
            row[k] = orcell(P, c, r, kind)
        ss = po[(po.set == P) & (po.Compare == c) & (po.Reference == r) & po.kind.isin(SENS + ["S.A.-corrected, D27 gated"])]
        row["range"] = f"{ss.OR.min():.2f}–{ss.OR.max():.2f}" if len(ss) else "–"
        gr = po[(po.set == P) & (po.Compare == c) & (po.Reference == r) & (po.kind == "S.A.-corrected, D27 gated")]
        row["own"] = (f"{int(gr.iloc[0].k_own)} of {int(gr.iloc[0].k)}; largest weight {100 * gr.iloc[0].max_weight:.0f}% "
                      f"({gr.iloc[0].max_weight_exp})") if len(gr) and pd.notna(gr.iloc[0].get("max_weight")) else "–"
        rows.append(row)
        rowsB.append({"pair": pair_txt(c, r), "MRID": orcell("B", c, r, "MRID (uncorrected)"),
                      "gated": orcell("B", c, r, "S.A.-corrected, D27 gated")})
    f17 = F["17_count_synthesis"]
    vc1 = vc[vc.set == P]
    u = po[(po.set == P) & (po.kind == "MRID (uncorrected)")].set_index(["Compare", "Reference"])
    gg = po[(po.set == P) & (po.kind == "S.A.-corrected, D27 gated")].set_index(["Compare", "Reference"])
    fz = po[(po.set == P) & (po.kind == "S.A.-corrected, D26 as frozen")].set_index(["Compare", "Reference"])
    xdp = [(c, r) for c, r in PAIRS17 if c.startswith("XDP")]
    d26, d27 = dec[P]["D26 as frozen"], dec[P]["D27 gated"]
    pe17 = R.step_csv("17_count_synthesis", "or_corrected_per_experiment.csv")
    pg17 = pe17[(pe17.set == P) & (pe17.version == "D27 gated") & (pe17.capture_ratio_source == "own S.A. wells")]
    own_txt = "; ".join(f"{pair_txt(c, r)}: {', '.join(e for e in EXPS if e in set(pg17[(pg17.Compare == c) & (pg17.Reference == r)].exp)) or 'none'}"
                        for c, r in PAIRS17)
    vmax = vc1.loc[vc1.implied_if_seen_like_SA.idxmax()]
    one_img = gsc[gsc.exp.isin(["XDP3", "XDP11"])]
    import re as _re
    b09 = po[(po.set == "B") & (po.kind == "S.A.-corrected, D27 gated") & (po.Compare == "XDP-33109.2B")
             & (po.Reference == "CTR-33114.C")]
    b09_txt = ""
    if len(b09) and (b09.iloc[0].lo > 1 or b09.iloc[0].hi < 1):
        x = b09.iloc[0]
        mloo = _re.search(rf"without {x.max_weight_exp}: ([^;]+)", str(x.loo))
        bv = po[(po.set == "B") & (po.Compare == "XDP-33109.2B") & (po.Reference == "CTR-33114.C")
                & po.kind.str.startswith("S.A.-corrected, D27 gated")]
        b09_txt = (f"Under track 2 the gated correction gives 09B vs 14C {x.OR:.2f} [{x.lo:.2f}, {x.hi:.2f}], but this rests on "
                   f"{x.max_weight_exp} ({100 * x.max_weight:.0f}% of the weight; 2 edge-column S.A. wells with Poisson-only "
                   f"intervals): without it {mloo.group(1) if mloo else '–'}, and across readouts {bv.OR.min():.2f}-{bv.OR.max():.2f}. "
                   "Under track 2 the capture fraction also mixes dropout with cells left in rule B's dropped band, so this is "
                   "not a visibility correction. ")
    c2d = c2[c2.fit == P]
    def c2_pool(cmp_, ref_, drop=None):
        g = c2d[(c2d.Compare == cmp_) & (c2d.Reference == ref_) & (c2d.exp != drop)]
        m = dl_pool(g.est, g.se)
        return f"{np.exp(-7 * m['mu']):.2f} [{np.exp(-7 * m['hi']):.2f}, {np.exp(-7 * m['lo']):.2f}]"
    a63 = g1(c1, P, "XDP-33363.D", "CTR-33114.C")
    a63n = g1(c1, f"{P}_noXDP10", "XDP-33363.D", "CTR-33113.2I")
    n63 = int((-c2d[(c2d.Compare == "XDP-33363.D") & (c2d.Reference == "CTR-33114.C")].est > 0).sum())
    k63 = int(((c2d.Compare == "XDP-33363.D") & (c2d.Reference == "CTR-33114.C")).sum())
    txt63 = (f"Live-cell counts: in the mixed model 63D keeps fewer live cells than 14C ({wk(a63.ratio_per_day)} "
             f"[{wk(a63.upper)}, {wk(a63.lower)}] per week; {n63} of {k63} experiments point that way), and fewer than 13I only "
             f"with XDP10 (without it {wk(a63n.ratio_per_day)} [{wk(a63n.upper)}, {wk(a63n.lower)}]). Pooled from the "
             f"per-experiment fits with small-sample intervals the 63D-vs-14C interval includes 1 "
             f"({c2_pool('XDP-33363.D', 'CTR-33114.C')}; without XDP1 {c2_pool('XDP-33363.D', 'CTR-33114.C', 'XDP1')}). "
             "These are net live-cell differences, and counts and odds ratios disagree too often for either to stand alone. ")
    ev17 = f"""<p><b>Decision outcome, {PL}.</b> D26 as frozen: visibility differs between lines in
{', '.join(d26['failing']) or 'no experiment'}, so by that rule the S.A.-corrected odds ratios would be primary. With the
validity check (D27, proposed): differs in {', '.join(d27['failing']) or 'no experiment'}; no valid readout in
{', '.join(d27['no_readout'] + ['XDP1', 'XDP8']) }. With or without the gate, the frozen D26 rule makes the corrected
odds ratios primary; D27 recommends keeping the MRID odds ratios primary instead, a change made after the results were
seen and logged as such. Both are shown; no version changes the track-1 conclusion below.</p>
<h3>17.1 Pooled odds ratios with and without the dropout correction ({PL})</h3>
{R.table(pd.DataFrame(rows), ["pair", "MRID", "corrected, D26 as frozen", "corrected, gated (D27)", "own", "range"],
         ["Pair", "Uncorrected: per-experiment MRID odds ratios, pooled [95% CI]", "Corrected, D26 as frozen",
          "Corrected, with the gate (D27)", "Gated: experiments with their own S.A. readout; largest weight",
          "Corrected, range over readouts (gated; XDP10 at T9; XDP11 baseline T11)"])}
<p>Per experiment: Step 13's time-averaged odds ratio multiplied by the ratio of the two lines' capture fractions
(reference ÷ comparison). Experiments without a valid S.A. readout for the pair use the capture ratio pooled over
experiments ({own_txt}); its spread between experiments is added to each, and its own estimation error once to the pooled
result. With only 2-3 contributing experiments that ratio is itself uncertain, and its covariance with the experiments
that supply it is ignored, so the corrected intervals are if anything too narrow. Pooled with a random-effects model and
small-sample (Hartung-Knapp) intervals - wider than the meta-analysis intervals of Step 13, which used normal quantiles.
"↑ / ↓": experiments whose own 95% interval lies above / below 1.</p>
<p>Track 2 (rule B) for comparison - its capture fraction also counts cells left in rule B's dropped band, so its correction
is not a visibility correction:</p>
{R.table(pd.DataFrame(rowsB), ["pair", "MRID", "gated"], ["Pair", "Track 2, MRID", "Track 2, corrected with the gate"])}
{R.figure(f17['F17_1'], 'Figure 17.1')}
<h3>17.2 Do odds ratios and live-cell loss agree?</h3>
<p>Per line pair × experiment ({PL}), on exactly the same wells and timepoints, the dead/live odds ratio and the
live-cell loss point the same way in {agr['sign_agree']} of {agr['n']} cases (Spearman ρ = {agr['spearman']:.2f}); where both
are clearly different from no difference, in {agr['sign_agree_both_clear']} of {agr['n_both_clear']}. Against MRID's
whole-run odds ratios: {agr['sign_agree_mrid']} of {agr['n_mrid']}; after the gated S.A. correction: {agr['sign_agree_corr']}
of {agr['n_mrid']}. Track 2: {ag['B']['sign_agree']} of {ag['B']['n']}. (The cases share lines, so they are not independent
and no p-value is attached.) If both readouts tracked death they would agree almost every time. They agree at chance level,
so in most experiments at least one of them is not measuring the death difference - the counts through line-specific
gains and losses of cells, or the odds ratios through visibility or cutoff behaviour - and the data alone cannot say
which.</p>
{R.figure(f17['F17_2'], 'Figure 17.2') if 'F17_2' in f17 else '<p>No experiment both passes the calibration test and shows no capture difference between lines, so no dropout-corrected cumulative death is shown.</p>'}
<h3>17.3 How much death is invisible? (S.A.-based, order of magnitude only)</h3>
<p>If spontaneous death were seen the way S.A. death is, the share of untreated cells that died within the last one to two
imaging intervals (24-48 h) would be the visible dead share divided by c. At the S.A. readout ({PL}, cases passing the
gate): visible {rng_txt(100 * vc1.visible_dead_share_ut.min(), 100 * vc1.visible_dead_share_ut.max(), 1)}%, implied
{rng_txt(100 * vc1.implied_if_seen_like_SA.min(), 100 * vc1.implied_if_seen_like_SA.max(), 1)}% (median
{100 * vc1.implied_if_seen_like_SA.median():.1f}%; highest {vmax.exp} {SHORT[vmax.line]}). One image after dosing only
{rng_txt(100 * one_img.c.min(), 100 * one_img.c.max(), 0)}% of S.A.-killed cells are still seen as dead (XDP3, XDP11), while the
calibration test puts the visible time of spontaneously dead cells at {', '.join(f"{r.V_hours:.0f} h ({r.exp})" for r in j1d[j1d.passes.astype(bool)].itertuples()) or 'unknown'};
if spontaneous dead cells stay visible longer than S.A.-killed ones, the upper end of this range is too high.</p>"""
    take17 = (f"<p>Correcting for dropout ({PL}, with the gate) moves the pooled XDP-vs-control odds ratios from "
              + "; ".join(f"{pair_txt(*k)} {u.loc[k].OR:.2f} to {gg.loc[k].OR:.2f}" for k in xdp)
              + (lambda nu, ng, nf: ". The 95% interval includes 1 for every XDP-vs-control comparison, before correction, with "
                 "the gated correction and with D26 as frozen. " if nu + ng + nf == 0 else
                 f". The 95% interval excludes 1 in {nu} of {len(xdp)} XDP-vs-control comparisons before correction, {ng} with "
                 f"the gated correction and {nf} with D26 as frozen. ")(
                  sum(u.loc[k].lo > 1 or u.loc[k].hi < 1 for k in xdp), sum(gg.loc[k].lo > 1 or gg.loc[k].hi < 1 for k in xdp),
                  sum(fz.loc[k].lo > 1 or fz.loc[k].hi < 1 for k in xdp))
              + b09_txt + txt63
              + "<b>Bottom line:</b> GFP dropout hides a large share of S.A. death (and probably of spontaneous death) in "
              "absolute terms; whether it biases the comparison between lines cannot be settled with these data, and no "
              "track-1 version of the analysis shows a reproducible XDP-vs-control difference in death.</p>")
    out.append(step_block(
        "s17", "17", "Counts and GEDI together",
        "Does accounting for GFP dropout change the comparison between lines?",
        "<p>The decision rules (D26 as frozen; D27 proposed), the S.A.-corrected odds ratios in every version, and the "
        "agreement between odds ratios and live-cell loss.</p>", ev17, take17, decision_refs(["D24", "D25", "D26", "D27"], decs)))
    return "\n".join(out)


def build(label: str) -> str:
    decs = cfg.decisions()
    dec_by = {d["id"]: d for d in decs}
    exps_cfg = cfg.experiments()
    params = cfg.params()
    k_sd = params["cutoff_rule"]["k_sd"]
    S = {s: R.step_json(s, "summary") for s in STEPS}
    F = {s: R.figs_of(s) for s in STEPS}
    checks = R.all_checks(STEPS)
    pp_all = R.step_csv("05_prepost", "prepost_results.csv")
    pp = pp_all[pp_all.baseline == "primary"]
    model = R.step_csv("05b_model_check", "model_check.csv")
    pop = R.step_csv("05c_cutoff_preview", "populations.csv")
    rules = R.step_csv("05c_cutoff_preview", "rule_consequences.csv")
    pattern = R.step_csv("05_prepost", "shift_pattern.csv")
    audit = pd.read_csv(RAW_CACHE / "MRID-rework/audit/threshold_audit_ALL.csv")
    staging = json.loads((RAW_CACHE / "staging_manifest.json").read_text())
    extract = S["01b_extract"]
    md, x10, rep, dt = S["01_metadata"], S["02a_xdp10_verify"], S["02c_crosscheck"], S["04_dose_timing"]
    dup = md["XDP10"]["duplicate_block"]

    # ---------------- derived numbers (everything quoted in prose is computed here) ----------------------
    import pyarrow.parquet as pq
    inscope = {}
    lt002 = {}
    for e in EXPS:
        t = pq.read_table(str(Path(f"/Users/aholub/xdp_rgedi_cache/parquet/{e}.parquet")),
                          columns=["well", "col", "ratio", "in_scope", "condition", "tp", "line"]).to_pandas()
        s_ = t[t.in_scope]
        inscope[e] = {"cells": len(s_), "wells": s_.well.nunique(), "nonpos": int((s_.ratio <= 0).sum()),
                      "x3_removed_present": int(t[t["col"].between(2, 5)].well.nunique()) if e == "XDP3" else None}
        if e in ("XDP1", "XDP3", "XDP5", "XDP10", "XDP11"):
            fit = {"XDP1": [13, 14], "XDP3": [10], "XDP5": [9], "XDP10": [9, 10], "XDP11": [13]}[e]
            sa = s_[(s_.condition == "sa") & s_.tp.isin(fit)]
            lt002[e] = float((sa.ratio < 0.02).mean())
    n_cells = sum(v["cells"] for v in inscope.values())
    input_steps = ["01_metadata", "01b_extract", "02a_xdp10_verify", "02c_crosscheck", "03_qc_counts"]
    ic = checks[checks.step.isin(input_steps)]
    rc = checks[~checks.step.isin(input_steps)]
    n_in_fail = int((ic.status == "FAIL").sum())

    med = pp[pp.stat == "med"].copy()
    last_med = med.sort_values("post_tp").groupby(["exp", "line"]).tail(1)
    ctl = last_med[last_med.exp != "XDP1"].copy()
    ctl["rose"] = np.where(ctl.few_wells.astype(bool), (ctl.perm_p < 0.05) & (ctl.fold > 1), ctl.did_lo > 0)
    rose = ctl[ctl.rose]
    not_rose = ctl[~ctl.rose]
    cnt = pp[pp.stat == "count"].sort_values("post_tp").groupby(["exp", "line"]).tail(1)
    gfp = pp[pp.stat == "gfp"].sort_values("post_tp").groupby(["exp", "line"]).tail(1)
    rfp = pp[pp.stat == "rfp"].sort_values("post_tp").groupby(["exp", "line"]).tail(1)
    gfp_rel = gfp[gfp.exp != "XDP1"].assign(rel=lambda x: 100 * (10 ** (x.delta_sa - x.delta_ut) - 1))
    rfp_rel = rfp[rfp.exp != "XDP1"].assign(rel=lambda x: 100 * (10 ** (x.delta_sa - x.delta_ut) - 1))
    gfp_fell = gfp_rel[gfp_rel.rel < 0]
    rfp_flat = rfp_rel[rfp_rel.rel <= 2]
    dsum = S.get("05d_drift_preview", {})

    def cnt_range(es):
        c = cnt[cnt.exp.isin(es)].count_ratio
        return (c.min(), c.max())
    x1 = pp[(pp.exp == "XDP1") & (pp.stat == "med")].sort_values("post_tp").groupby("line").tail(1)
    x1_pass = int((x1.ratio_to_comparison_max > 3).sum())
    x1s = dt.get("XDP1", {})
    t67 = x1s.get("t6_t7_median_change", {})
    x1_t11 = x1s.get("t10_t11_ratio_median_log", {})

    # Shift patterns
    pat_counts = pattern[pattern.exp != "XDP1"].pattern.value_counts().to_dict()

    # Preview
    sp = S.get("05c_cutoff_preview", {})
    pop["separated"] = pop.dead_found.astype(bool) & (pop.ashman_ut_dead >= params["usability"]["min_ashman_d"])
    sep_keys = set(zip(pop[pop.separated].exp, pop[pop.separated].line))
    rules = rules[[k in sep_keys for k in zip(rules.exp, rules.line)]]
    rB = rules[rules.rule == "B"]
    rC = rules[rules.rule == "C"]
    nB_above = sp.get("rule_B_dead_cutoff_above_dead_centre", 0)
    overlap = sp.get("two_cutoff_bands_overlap_cases", 0)
    found = pop[pop.separated]

    # Usability (R4) per case from the preview: dead found, Ashman D >= 2, dead share >= 10 points above
    # the untreated share above the rule-C cutoff, and >= 2 S.A. wells.
    def usable(exp, line):
        r = pop[(pop.exp == exp) & (pop.line == line)]
        if not len(r) or not bool(r.dead_found.iloc[0]):
            return False, "no dead population (3-comp, ≥ 5%)"
        r = r.iloc[0]
        excess = r.dead_excess
        if r.ashman_ut_dead < params["usability"]["min_ashman_d"]:
            return False, f"overlaps live (D = {r.ashman_ut_dead:.1f})"
        if not excess >= params["usability"]["min_sa_dead_excess"]:
            return False, f"dead share {100 * r.dead_w:.0f}% vs {100 * r.ref_dead_w:.0f}% in reference (< 10 points more)"
        return True, f"dead {100 * r.dead_w:.0f}% vs {100 * r.ref_dead_w:.0f}% reference, at ratio {r.dead_mu:.2f}; D = {r.ashman_ut_dead:.1f}"

    n_sa_cases = len(pop)
    n_usable = sum(usable(r.exp, r.line)[0] for r in pop.itertuples())

    # XDP11 edge check: column 22 vs columns 17-21 (both S.A.), per-well median change T12 -> T13.
    wo = R.step_csv("04_dose_timing", "well_outcomes_XDP11.csv")
    edge_rows = []
    if len(wo):
        wo["col"] = wo.well.str[1:].astype(int)
        pv = wo[wo.condition == "sa"].pivot_table(index=["well", "line", "col"], columns="tp", values="median_log").reset_index()
        pv["chg"] = pv[13] - pv[12]
        for line in figs.LINE_ORDER:
            g = pv[pv.line == line]
            if len(g):
                edge_rows.append({"line": SHORT[line], "c22": f"{10 ** g[g.col == 22].chg.mean():.2f}×",
                                  "c1721": f"{10 ** g[g.col < 22].chg.mean():.2f}×"})

    toc = [("decide", "What needs deciding"), ("summary", "Summary"), ("limits", "Known limitations"),
           ("status", "Status by line and experiment"), ("changes", "What changed, and why"),
           ("s1", "1 · Scope and design"), ("s2", "2 · Data integrity"), ("s3", "3 · When S.A. took effect"),
           ("s4", "4 · Before vs after S.A."), ("s5", "5 · Rules, and what the data forced"),
           ("s6", "6 · Cutoffs from S.A."), ("s6b", "6b · Buffer zone (run 2)"), ("s7", "7 · Do they hold?"), ("s8", "8 · Transfer"),
           ("s9", "9 · Sensitivity"), ("s10", "10 · Final thresholds"), ("s11", "11 · Live/dead calls for MRID"),
           ("s12", "12 · MRID log-odds"), ("s13", "13 · Odds ratios across lines"), ("s14", "14 · Cell counts over time"),
           ("s15", "15 · How much S.A. death GEDI sees"), ("s16", "16 · Live-cell retention"),
           ("s17", "17 · Counts and GEDI together"), ("decisions", "Decision log"), ("checks", "All checks"),
           ("glossary", "Glossary"), ("provenance", "Provenance")]
    out = ["<title>XDP RGEDI Threshold Evidence</title>", FONTS, f"<style>{CSS}</style>", '<div class="page">',
           '<nav class="toc" aria-label="Steps"><ol>' + "".join(f'<li><a href="#{a}">{E(t)}</a></li>' for a, t in toc)
           + "</ol></nav><main>"]
    out.append(f"""<header class="title"><span class="eyebrow">Finkbeiner lab · XDP iPSC neurons · RGEDI survival</span>
<h1>XDP RGEDI threshold evidence</h1>
<p class="lede">Live/dead GEDI-ratio cutoffs for each cell line in each experiment, calibrated on cells killed with sodium
arsenite (S.A.). This is the step-by-step record of how they are derived, written to be read in order.</p>
<div class="meta"><span>Version <b>{E(label)}</b> · analysis run <b>{E(RUN)}</b></span>
<span>Built {time.strftime('%Y-%m-%d %H:%M')}</span><span>Prepared by Austin Holub with Claude</span><span>For: Julia Kaye</span></div></header>""")

    # ---------------- what needs deciding -------------------------------------------------------------------
    def dline(i):
        d = dec_by[i]
        return f'<li><a href="#{i}"><b>{i}</b></a> {E(str(d["question"]))}</li>'
    open_ids = [d["id"] for d in decs if d["status"] == "proposed"]
    done_a = [i for i in ["D09", "D10", "D11", "D12", "D14", "D16", "D18", "D19", "D20"] if dec_by.get(i, {}).get("status", "").startswith("approved")]
    out.append(f"""<section class="step" id="decide"><div class="decide"><h4>What needs deciding at this review point</h4>
<ul class="summary-list">{''.join(dline(i) for i in open_ids)}</ul>
<p>Decided at review point A (2026-10-13): {', '.join(f'<a href="#{i}">{i}</a>' for i in done_a)} - including running two
threshold tracks side by side (D20): <b>track 1</b> = rule D (equal-density boundary), with rule C as its sensitivity;
<b>track 2</b> = rule B (2.5 SD on both sides). Steps 1–5 are the evidence behind those decisions; Steps 6–10 compute
the thresholds for both tracks.</p>
<p><b>Run 2 (2026-10-15, D28):</b> the cutoff is now the rule D centre with a buffer zone of undecided cells, set where a
cell becomes 90% likely to be live or dead (Step 6b). Every result from Step 11 on uses this {E(PL)}, with the run-1 single
cutoff and the 95% window as comparisons. Run 1 is kept unchanged in its own folder and document.</p></div></section>""")

    # ---------------- summary ---------------------------------------------------------------------------
    sept_lines = int((audit.baseline_death_pct.round(1) == 9.1).sum())
    _po = R.step_csv("13_or_figures", "pooled_or_by_set.csv")
    or_bullet = ""
    if len(_po) and "lxe_OR" in _po:
        _x = _po[_po.Compare.str.startswith("XDP") & ~_po.set.str.endswith("_shared")]
        _a = _x[_x.Reference == "CTR-33114.C"]
        _incl = int(((_a.lxe_lo <= 1) & (_a.lxe_hi >= 1)).sum())
        _mex = _a[(_a.meta_all_lo > 1) | (_a.meta_all_hi < 1)]
        _mtxt = ("" if not len(_mex) else f" (random-effects meta-analysis: {len(_a) - len(_mex)} of {len(_a)})")
        or_bullet = (f"<li><b>Odds ratios (Step 13).</b> MRID's combined model, which treats each line difference as the same "
                     f"in every experiment, shows more death in both XDP lines than in control 14C (pooled odds ratio "
                     f"{rng_txt(_a.mridpool_OR.min(), _a.mridpool_OR.max())}). But the per-experiment differences vary far "
                     f"more than that model allows; once that variation is included, the 95% interval for XDP vs 14C contains "
                     f"1 in {_incl} of {len(_a)} threshold-set × line estimates{_mtxt}. Against control 13I the XDP lines do not show "
                     "more death. The data do not show a reproducible XDP-vs-control difference in death (D24, D25).</li>")
    cnt_bullet = ""
    if S.get("17_count_synthesis"):
        _cap = R.step_csv("15_sa_capture", "capture.csv")
        _c = _cap[(_cap.set == "D_sd") & (_cap.variant == "primary") & _cap.valid.astype(bool)]
        _q = R.step_csv("17_count_synthesis", "or_pooled_uncorrected_vs_corrected.csv")
        _x = _q[(_q.set == P) & _q.Compare.str.startswith("XDP")]
        _ex = lambda kind: int(((_x.kind == kind) & ((_x.lo > 1) | (_x.hi < 1))).sum())  # noqa: E731
        _n = int((_x.kind == "MRID (uncorrected)").sum())
        _ag = S["17_count_synthesis"]["agreement"][P]
        cnt_bullet = (f"<li><b>Cell counts and GFP dropout (Steps 14–17, {PL}).</b> Until Step 14 the death estimate did not "
                      "use cell counts. The S.A. wells show that at the S.A. readout GEDI sees only "
                      f"{rng_txt(_c.c.min(), _c.c.max())} of the killed cells as dead (measured with the single cutoff); the rest "
                      "are no longer detected. If "
                      "spontaneously dying cells drop out in the same way, MRID's dead share understates death in every line; "
                      "that biases a comparison between lines only if the lines differ in it, and the data cannot settle that: "
                      "the S.A. capture fraction changes with the readout timepoint, and live-cell counts and GEDI odds ratios "
                      f"point the same way in only {_ag['sign_agree']} of {_ag['n']} line pairs × experiments. Live-cell counts "
                      "alone suggest faster net loss of live cells in 63D than in 14C in the mixed model, but not once "
                      "per-experiment estimates are pooled with small-sample intervals. "
                      + ("With or without the dropout correction (D26 as frozen, or with the validity check proposed in D27), "
                         "the 95% interval of every XDP-vs-control odds ratio includes 1."
                         if _ex("MRID (uncorrected)") + _ex("S.A.-corrected, D26 as frozen") + _ex("S.A.-corrected, D27 gated") == 0
                         else f"The 95% interval of the XDP-vs-control odds ratio excludes 1 in {_ex('MRID (uncorrected)')} of "
                              f"{_n} comparisons without correction, {_ex('S.A.-corrected, D26 as frozen')} with D26 as frozen "
                              f"and {_ex('S.A.-corrected, D27 gated')} with the validity check proposed in D27.")
                      + "</li>")
    out.append(f"""<section class="step" id="summary"><h2>Summary</h2><ul class="summary-list">
<li><b>What this is.</b> A redo of the live/dead thresholds for {len(EXPS)} XDP experiments and 4 cell lines (14C and
13I healthy controls; 09B and 63D XDP), using cells that S.A. killed to locate the dead population. {n_cells:,} segmented
cells are in scope.</li>
<li><b>Why.</b> The September rule (live = median, dead = 95th percentile of each line's T1 cells) sets T1 death to
0.05 / 0.55 = 9.1% by construction - it is 9.1% in {sept_lines} of {len(audit)} line × experiment rows of the September
audit - and S.A. was not used to set it. Re-checking the inputs also found data errors that affected earlier runs (Step 2).</li>
<li><b>Inputs.</b> {len(ic)} input checks, {n_in_fail} failures. XDP10's per-cell file was truncated; a complete
re-extract is identical to it on all {x10['overlap']['groups_compared']:,} overlapping well × timepoint × channel groups
and restores the missing wells. Re-running MRID's ratio code on XDP8's raw data reproduces the September file byte for byte.</li>
<li><b>Before vs after S.A. (Step 4).</b> In {len(rose)} of {len(ctl)} line × experiment cases that have untreated wells,
S.A. raised the median GEDI ratio more than the untreated wells changed over the same interval (fold
{rng_txt(rose.fold.min(), rose.fold.max())}); the exception{'s are' if len(not_rose) > 1 else ' is'}
{', '.join(f'{r.exp} {SHORT[r.line]}' for r in not_rose.itertuples())}. Part of the rise is GFP loss: relative to untreated
wells, median GFP in S.A. wells fell in {len(gfp_fell)} of {len(gfp_rel)} cases (by {rng_txt(-gfp_fell.rel.max(), -gfp_fell.rel.min(), 0)}%),
and in {', '.join(f'{r.exp} {SHORT[r.line]}' for r in rfp_flat.itertuples())} RFP did not rise beyond untreated at all.
XDP1 (no untreated wells) does not pass its own advance check for the first dose (Step 3).</li>
<li><b>What the data changed.</b> Three planned rules did not fit the data and are now open decisions: how the dead
population is found (D18), what counts as an S.A. onset (D19), and the cutoff rule itself (D20): the untreated live
population is too broad on the log scale for "mean ± 2.5 SD" to be applied on both sides. {n_usable} of {n_sa_cases} S.A. cases
have a usable dead population; the others, and XDP8, will be transferred. The cutoff options differ mainly in how they
treat the upward drift of living cells late in a run (Step 5.3).</li>
{or_bullet}{cnt_bullet}</ul></section>""")

    # ---------------- limitations ---------------------------------------------------------------------------
    c1011 = cnt_range(["XDP10", "XDP11"])
    out.append(f"""<section class="step" id="limits"><h2>Known limitations</h2><ul class="summary-list">
<li><b>Segmentation finds cells by their GFP (FITC) signal.</b> Dying cells lose GFP and can drop out: S.A. wells kept
{rng_txt(*c1011)} of the cells that untreated wells kept in XDP10 and XDP11. The S.A.-dead population we can measure is the
part that is still detectable. Steps 14–17 measure this dropout for S.A.-killed cells and test whether it differs between
lines; the test is inconclusive and the S.A. correction depends on the readout, so the corrected odds ratios are shown as a
sensitivity range (D27, pending), not as a replacement.</li>
<li><b>S.A. death is not disease death.</b> The cutoffs only need to locate the dead state of the sensor (RGEDI reports
the calcium rise shared by dying cells); whether naturally dying cells in untreated wells sit at the same ratio is
checked in Step 7.</li>
<li><b>"Fixed in advance" means fixed by Austin on 2026-10-13, before the threshold fits</b> - an internal plan, not an
external pre-registration, and Julia has not yet reviewed it. Exploratory mixture fits on XDP5 were run before the
deviations D18-D20 were proposed; they are described where used.</li>
<li><b>XDP10</b> has 2 S.A. wells per line, all in the outermost used column (22): its intervals can only redraw cells
within those wells (they understate uncertainty), and position cannot be separated from treatment after dosing. <b>XDP3</b>'s dose time is inferred. <b>XDP1</b> has no
untreated wells.</li></ul></section>""")

    # ---------------- status ---------------------------------------------------------------------------
    rows = []
    for exp in EXPS:
        roles = cfg.well_roles(exp)
        for line in figs.LINE_ORDER:
            if line not in exps_cfg[exp]["lines"]:
                continue
            n_sa = int(((roles.line == line) & (roles.condition == "sa")).sum())
            m = last_med[(last_med.exp == exp) & (last_med.line == line)]
            c = cnt[(cnt.exp == exp) & (cnt.line == line)]
            if exp == "XDP8":
                route = R.badge("FAIL", "transfer: no S.A. images")
            else:
                ok, why = usable(exp, line)
                route = R.badge("PASS" if ok else "WARN", ("usable: " if ok else "transfer: ") + why)
            onset = {"XDP1": "T11 / T13 (record)", "XDP3": "T10 (inferred)", "XDP5": "T8 (record)", "XDP8": "–",
                     "XDP10": "T9 (record)", "XDP11": "T13 (record)"}[exp]
            if len(m) and exp == "XDP1":
                shift = f"{10 ** m.delta_sa.iloc[0]:.2f} (no untreated)"
            elif len(m):
                mm = m.iloc[0]
                shift = (f"{mm.fold:.2f}, p {pfmt(mm.perm_p)} (2 wells)" if mm.few_wells
                         else f"{mm.fold:.2f} [{mm.fold_lo:.2f}, {mm.fold_hi:.2f}]")
            else:
                shift = "–"
            rows.append({"exp": exp, "line": SHORT[line], "n_sa": n_sa, "onset": onset, "shift": shift,
                         "count": f"{c.count_ratio.iloc[0]:.2f}" if len(c) else "–", "route": route})
    out.append(f"""<section class="step" id="status"><h2>Status by line and experiment</h2>
<p>One row per cell line in each experiment (21 in all). <b>Median shift</b>: fold change in the median ratio in S.A. wells
beyond the change in untreated wells (95% CI; Step 4). <b>Cells after S.A.</b>: detected cells in S.A. wells after vs
before dosing, divided by the same ratio in untreated wells (1 = no loss). <b>S.A. dead population</b>: whether the
post-dose S.A. cells contain a dead population that passes the usability rules fixed in advance (Step 5); if not, that
case's cutoffs will come from transfer (D03).</p>
{R.table(pd.DataFrame(rows), ["exp", "line", "n_sa", "onset", "shift", "count", "route"],
         ["Exp", "Line", "S.A. wells", "S.A. onset", "Median shift (fold)", "Cells after S.A.", "S.A. dead population"],
         num_cols=["n_sa", "count"], raw_cols=["route"])}</section>""")

    # ---------------- what changed ---------------------------------------------------------------------------
    dbl = R.step_csv("01_metadata", "xdp10_platemap_duplicate_block.csv")
    relab = dbl[(dbl.appended_drug == "SA") & (dbl.appended_line == "CTR-33362.C")]
    relab_txt = ", ".join(f"{r.well} (first block: {SHORT.get(r.first_block_line, r.first_block_line)})" for r in relab.itertuples())
    fr = audit.groupby("experiment")[["firstrun_live_global", "firstrun_dead_global"]].first()
    fr_txt = "; ".join(f"{e} {r.firstrun_live_global:g}/{r.firstrun_dead_global:g}" for e, r in fr.iterrows() if e in EXPS)
    ch = [
        ("First run (Sept 4)", f"One live/dead pair per experiment, read by eye from MRID plots ({fr_txt}).",
         "Not reproducible; ignores differences between lines; not tied to known-dead cells.",
         "A written rule applied per line and experiment, calibrated on S.A.-killed cells."),
        ("September rework rule", "Live = median, dead = 95th percentile of each line's own T1 cells.",
         f"T1 death is fixed at 9.1% by construction (9.1% in {sept_lines} of {len(audit)} audit rows), so T1 differences "
         "between lines are erased; 45% of T1 cells fall between the cutoffs and are dropped.",
         "T1 death becomes a measured output."),
        ("Use of S.A.", "Checked afterwards as pass/fail.",
         "S.A. did not inform the cutoffs; the pass/fail bar was applied unevenly (33362.C dropped, XDP3 09B kept).",
         "S.A.-killed cells locate the dead population; a case without a usable dead population is transferred, not dropped."),
        ("XDP10 per-cell file", "Read GXYTMP/cell_data.csv; noted a 'malformed footer'.",
         f"The file stops mid-row ({x10['truncation']['bytes']:,} bytes): {x10['missing_from_original']['groups']} well × "
         "timepoint × channel groups were missing (all of row M = 63D, and L22 after T1).",
         "Verified complete re-extract (Step 2)."),
        ("XDP10 plate map", "Merged a plate map that has an appended second block.",
         f"{dup['wells_also_in_first_block']} wells were listed twice ({dup['label_differs']} with a different line), so their "
         f"cells were counted twice. The appended block labelled {relab_txt} as 33362.C S.A., so the September '33362.C S.A. "
         "failure' was measured partly on another line's well.",
         "First block only; S.A. = column 22, rows D–M (Step 2)."),
        ("XDP3 well removal", "Config asked to remove columns 2–5.",
         f"Well names did not match ('C2' vs 'C02'), so nothing was removed: all {inscope['XDP3']['x3_removed_present']} wells of "
         "columns 2–5 are in the September file.", "Well roles come from one config checked against every plate map."),
        ("siRNA wells", "Pooled with each line's untreated wells.", "The CTR siRNA itself kills cells in XDP3 column 3.",
         "Untreated = no drug and no siRNA."),
        ("'noSA' comparison", "Removed S.A. wells and dropped 33362.C in one run.", "Two changes at once.",
         "One change per step in the odds-ratio phase."),
        ("T1 death fix", "Reported as changing results.",
         "It changed only the plotted T1 death; the model fits raw live/dead counts (MRID_Logodd_Functions_June2026.R).",
         "Retracted."),
        ("Info workbook", "Blocks labelled XDP8 and XDP10.", "The two labels are swapped (Step 2.4).", "Corrected."),
    ]
    trs = "".join(f"<tr><td><b>{E(a)}</b></td><td>{E(b)}</td><td>{E(c)}</td><td>{E(d)}</td></tr>" for a, b, c, d in ch)
    out.append(f"""<section class="step" id="changes"><h2>What changed, and why</h2>
<p>Kept from before: per-line thresholds (signal-to-noise differs by line), MRID's ratio (RFP mean / GFP mean per cell),
and the Galaxy segmentation itself. MRID's two-cutoff scheme is kept unless D20 replaces it with a single cutoff.</p>
<div class="table-wrap"><table><thead><tr><th>Item</th><th>What we did</th><th>Why it was a problem</th><th>What we do now</th></tr></thead>
<tbody>{trs}</tbody></table></div></section>""")

    # ---------------- Step 1 ---------------------------------------------------------------------------
    e_rows = []
    for exp in EXPS:
        e = exps_cfg[exp]
        s = e["sa"]
        m = md[exp]
        dose = "; ".join(d["note"] for d in s.get("doses", []))
        prov = s.get("provenance", "verified")
        fit = {"XDP1": "T13–T14 (D14)", "XDP3": "T10 (D12)", "XDP8": "transfer"}.get(
            exp, ", ".join(f"T{t}" for t in s["fit_tps"]))
        e_rows.append({"exp": exp, "lines": ", ".join(SHORT.get(l, l) for l in e["lines"] if l in figs.LINE_ORDER),
                       "tps": f"T{min(m['timepoints'])}–T{max(m['timepoints'])} ({int(min(m['hours']))}–{int(max(m['hours']))} h)",
                       "design": {"whole_plate": "whole plate", "wells": "S.A. wells", "none_imaged": "not imaged"}[s["design"]],
                       "dose": E(dose) + " " + tag(prov), "fit": fit})
    plates = "".join(f'<details><summary>{exp} plate map</summary><div class="inner">{PLATE_LEGEND}{plate_map(exp)}'
                     f'<p class="meta">{E(exps_cfg[exp].get("notes", ""))}</p></div></details>' for exp in EXPS)
    out.append(step_block(
        "s1", "1", "Scope and design",
        "Which cells, wells and timepoints enter the analysis, and when was S.A. added in each experiment?",
        "<p>All well roles (line; S.A., untreated or siRNA) and dosing facts come from one configuration file that is "
        "checked against every plate map (Step 2). Dosing facts are tagged " + tag("verified") + " (lab notebook and data "
        "agree) or " + tag("inferred") + " (derived from the data by a fixed rule). The S.A. concentration is recorded "
        "only for XDP5 (200 µM).</p>",
        R.table(pd.DataFrame(e_rows), ["exp", "lines", "tps", "design", "dose", "fit"],
                ["Exp", "Lines", "Imaging", "S.A. design", "Dose (source)", "S.A. dead population read at"], raw_cols=["dose"])
        + '<h3>Plate maps</h3>' + plates,
        "<p>6 experiments, 21 line × experiment cases. 14C, 09B and 63D appear in all six; 13I only in XDP5, XDP10 and XDP11. "
        "XDP8 was dosed but never imaged afterwards; XDP1 was dosed as a whole plate, so it has no untreated comparison wells.</p>",
        decision_refs(["D07", "D11", "D15"], decs)))

    # ---------------- Step 2 ---------------------------------------------------------------------------
    ex_rows = [{"exp": e, "source": "re-extract → MRID mrid_ratio()" if exps_cfg[e]["source"]["kind"] == "rebuild"
                else "September MRID per-cell file", "cells": f"{inscope[e]['cells']:,}", "wells": inscope[e]["wells"],
                "tps": ",".join(map(str, extract[e]["timepoints"])), "nonpos": inscope[e]["nonpos"]} for e in EXPS]
    blocks = pd.DataFrame(md["info_workbook_blocks"])
    ledger = []
    for e in EXPS:
        hr = R.step_csv("03_qc_counts", f"plate_highratio_{e}.csv")
        for r in hr[hr.spike].itertuples():
            ledger.append((e, f"T{r.tp}", f"{100 * r.frac_above_T1_p99:.1f}% of {'S.A.' if e == 'XDP1' else 'untreated'} "
                           f"cells above the T1 99th percentile ({r.fold_vs_T1_T3:.1f}× the T1–T3 average)"
                           + (" - after the doses, expected" if e == "XDP1" else " - plate-wide, not S.A.")))
    if t67:
        xd = R.step_csv("04_dose_timing", "xdp1_differences.csv")
        d1 = xd[(xd.outcome == "median_log") & (xd.t_to == 11)].set_index("line").mean_change
        bigger = [SHORT[l] for l, v in t67.items() if v > d1.get(l, np.inf)]
        ledger.append(("XDP1", "T6→T7", "median ratio jumps plate-wide by " + ", ".join(
            f"{SHORT[l]} {10 ** v:.2f}×" for l, v in t67.items()) + (f" (larger than the first S.A. dose in {', '.join(bigger)})"
            if bigger else "") + "; intervals touching T7 are left out of XDP1's pre-dose comparison"))
    t1 = R.step_csv("03_qc_counts", "totals_XDP1.csv").pivot_table(index="line", columns="tp", values="cells")
    ledger.append(("XDP1", "T10→T12", "detected cells change by " + ", ".join(
        f"{SHORT[l]} {100 * (t1.loc[l, 12] / t1.loc[l, 10] - 1):+.0f}%" for l in t1.index) + " after the first dose (unexplained)"))
    t11 = R.step_csv("03_qc_counts", "totals_XDP11.csv").groupby("tp").cells.sum()
    ledger += [("XDP5", "T4–T6", "well C06 has no cells"),
               ("XDP8", "T2", "removed by the September config, reason not recorded (D16)"),
               ("XDP11", "T11→T12", f"{100 * (1 - t11.loc[12] / t11.loc[11]):.0f}% fewer detected cells plate-wide, before the dose")]
    led = "".join(f"<tr><td>{a}</td><td class='mono'>{E(b)}</td><td>{E(c)}</td></tr>" for a, b, c in ledger)
    ev2 = f"""<h3>2.1 Inputs</h3>{R.table(pd.DataFrame(ex_rows), ["exp", "source", "cells", "wells", "tps", "nonpos"],
        ["Exp", "Per-cell source", "Cells (in scope)", "Wells (in scope)", "Timepoints", "Ratio ≤ 0"], num_cols=["cells", "wells", "nonpos"])}
<p>Checks on every experiment: one row per cell (timepoint, well, object); every cell maps to a configured well and a known
timepoint; the stored line label equals the configured line; ratio = RFP / GFP exactly. {int((ic.status == 'PASS').sum())}
pass, {int((ic.status == 'WARN').sum())} informational, {n_in_fail} fail (<a href="#checks">All checks</a>). Cells with
ratio ≤ 0 (RFP background-subtracted to 0) stay in for MRID but are left out of log-scale fits.</p>
<h3>2.2 XDP10 recovery</h3>
<p>The original <code>XDP10/GXYTMP/cell_data.csv</code> ends mid-row (last bytes <code>{E(x10['truncation']['partial_last_line'])}</code>).
The re-extract <code>rerun-ExtraCellInfo/cell_data.csv</code> was compared group by group:
<b>{x10['overlap']['groups_identical']:,} of {x10['overlap']['groups_compared']:,}</b> complete well × timepoint × channel groups
({x10['overlap']['rows_compared']:,} rows) are identical on every shared column; the cut-off group (L22, T1, RFP) has
{x10['overlap']['last_group']['rows_original']} rows in the original, all found among the {x10['overlap']['last_group']['rows_reextract']}
in the re-extract. The re-extract covers the full plate and restores {x10['missing_from_original']['groups']} groups. Per-cell ratios
were then recomputed with MRID's own <code>mrid_ratio()</code> on the corrected plate map.</p>
<h3>2.3 Reproducing the ratio step</h3>
<p>MRID's <code>mrid_ratio()</code> re-run today on XDP8's raw data with the September settings gives a file that is
<b>{'byte-identical' if rep['r_reproduction_xdp8']['byte_identical'] else 'NOT identical'}</b> to the September XDP8 file
({rep['r_reproduction_xdp8']['rows_a']:,} cells). An independent Python re-implementation gives the same cells and ratios
as MRID's R code on XDP8 and XDP10 (largest relative difference
{max(rep['python_port_XDP8']['max_rel_diff_ratio'], rep['python_port_XDP10']['max_rel_diff_ratio']):.0e}, CSV rounding).
XDP1, XDP3, XDP5 and XDP11 use the September files, which pass every internal check but were not re-run from raw data.</p>
<h3>2.4 Which workbook block is which</h3>
{R.table(blocks, ["workbook_label", "n_lines", "rows", "sa_columns", "matches_data_of", "note"],
         ["Workbook label", "Lines", "Rows", "S.A. cols", "Line set matches data of", "Note"])}
<p>The block labelled XDP8 has 5 lines in rows D–M and S.A. in column 22 - XDP10's layout; the block labelled XDP10 has
3 lines in rows C–N and no S.A. readout - XDP8's layout. The lab notebook agrees (XDP8: 3 lines, "imaging skipped (IXM
downtime) … plate tossed"; XDP10: 5 lines, "SA dose" on the T8 day).</p>
<h3>2.5 Timepoint ledger</h3><div class="table-wrap"><table><thead><tr><th>Exp</th><th>When</th><th>What</th></tr></thead>
<tbody>{led}</tbody></table></div>"""
    out.append(step_block(
        "s2", "2", "Data integrity and corrections",
        "Are the inputs complete, correctly labelled, and the same data MRID uses?",
        "<p>Every input was copied once from the share with a SHA-256 checksum, converted to a compact table and checked "
        "automatically. XDP10 was rebuilt from its complete re-extract; the ratio step was re-run on XDP8 to confirm it "
        "reproduces the September file.</p>", ev2,
        "<p>The September per-cell files are internally consistent, and the ratio step reproduces exactly where it was "
        "re-run; XDP10 now has complete data with one row per cell. The corrections change labels and completeness, not "
        "any measured value. Several plate-wide events (ledger) are not S.A. effects and are handled by comparing S.A. "
        "with untreated wells at the same timepoints.</p>",
        decision_refs(["D09", "D10", "D11", "D16"], decs),
        "<p>A re-extract group that differed from the original (different segmentation), a cell counted twice, or a line "
        "label disagreeing with the plate map. None occurred.</p>"))

    # ---------------- Step 3 ---------------------------------------------------------------------------
    t_rows = []
    for e in ["XDP3", "XDP5", "XDP10", "XDP11"]:
        s = dt[e]
        sup = s["boot_support"]
        nw = s["null_window"]
        t_rows.append({"exp": e, "doc": {"XDP3": "not recorded", "XDP5": "1 h before T8", "XDP10": "between T8 and T9",
                                          "XDP11": "after T12 imaging"}[e],
                       "q90": f"T{s['tau_hat']['q90_log']} ({100 * sup['q90_log']:.0f}%)",
                       "med": f"T{s['tau_hat']['median_log']} ({100 * sup['median_log']:.0f}%)",
                       "cnt": f"T{s['tau_hat']['log_count']} ({100 * sup['log_count']:.0f}%)",
                       "last": f"T{s['last_timepoint']}",
                       "rule": "yes" if s["d19_rule"]["pass"] else "no",
                       "null": f"T1–T{nw['last_tp']}: " + ("onset found" if nw["d19_rule_would_accept"]["q90_log"] else "none found")
                       + f" (support {100 * nw['boot_support']['q90_log']:.0f}%)"})
    ev_figs = "".join(f'<details><summary>{e} event study</summary><div class="inner">{R.figure(F["04_dose_timing"][f"Fevent_{e}"])}</div></details>'
                      for e in ["XDP3", "XDP5", "XDP10", "XDP11", "XDP1"] if f"Fevent_{e}" in F["04_dose_timing"])
    x1_first = x1s.get("first_interval_over_3x_q90_log", {})
    ev3 = f"""{R.table(pd.DataFrame(t_rows), ["exp", "doc", "q90", "med", "cnt", "last", "rule", "null"],
        ["Exp", "Dosing record", "Onset: 90th pct (bootstrap support)", "Onset: median", "Onset: cell count",
         "Last imaged", "Proposed rule accepts", "Same scan on pre-dose timepoints only"])}
<p>The 90th-percentile onset matches the record in XDP10 (T9) and XDP11 (T13); in XDP10 the median and cell count put the
onset one timepoint later (T10). In XDP5 the 90th percentile picks T7, the day before dosing, with only
{100 * dt['XDP5']['boot_support']['q90_log']:.0f}% bootstrap support, so the proposed rule rejects it; on T7 the S.A. wells
already sit above untreated wells, which we attribute to the plate-wide T7 event (ledger) but have not tested. The cell count
picks T3 and the median gives T{dt['XDP5']['tau_hat']['median_log']}. For XDP3 all three
outcomes give T10, but T10 is XDP3's last image, so the scan cannot place the onset later. Run on pre-dose timepoints only,
the proposed rule finds no onset in any experiment. XDP1 (no untreated wells): the T10→T11 change in median ratio is
{', '.join(f'{SHORT[l]} {v:.1f}×' for l, v in x1_t11.items())} the largest change between T1 and T10 (excluding T7), below
the 3× bar; the 12-hour 90th-percentile change first passes 3× at
{', '.join(f'{SHORT[l]} ' + (f'T{v}' if v else 'never') for l, v in x1_first.items())}.</p>{ev_figs}"""
    out.append(step_block(
        "s3", "3", "When did S.A. take effect?",
        "At which timepoint do S.A. wells first differ from untreated wells? Needed because XDP3's dose time was never "
        "recorded; checked on the experiments whose timing is recorded.",
        "<p>For each well and timepoint we take the median and the 90th percentile of log10 ratio and the cell count "
        "relative to T1. A regression with well and timepoint effects gives the S.A.-minus-untreated difference at each "
        "timepoint (the event-study figures). A scan over candidate onset timepoints picks the one that best separates "
        "before from after; redrawing wells 1,000 times (bootstrap) shows how stable that pick is. The plan also required "
        "the effect to exceed 3× the largest pre-dose difference in every line.</p>", ev3,
        "<p>The planned 3× rule rejects even XDP10 and XDP11, whose timing is recorded: it compares one post-dose effect with "
        "the largest of 7–11 noisy pre-dose differences, so it cannot discriminate (D19). The proposed rule recovers the "
        "recorded onset in 2 of 3 experiments (XDP10, XDP11; XDP5 is rejected, not mis-placed) and finds no onset on "
        "pre-dose-only timepoints. XDP3's "
        "onset is T10 " + tag("inferred") + ". XDP1 does not pass its own check for the first dose.</p>",
        decision_refs(["D19", "D12", "D13", "D14"], decs),
        "<p>An onset contradicting the record in XDP10 or XDP11, or the proposed rule accepting an onset on pre-dose "
        "timepoints. Neither happened. What did happen: XDP5's 90th percentile points at T7 (rejected by the support rule), "
        "XDP10's median and count onsets are one timepoint late, and XDP1's first-dose change is less than 3× its largest "
        "pre-dose change.</p>"))

    # ---------------- Step 4 ---------------------------------------------------------------------------
    p_rows = []
    for e in EXPS:
        if e == "XDP1":
            continue
        for line in figs.LINE_ORDER:
            g = pp[(pp.exp == e) & (pp.line == line)]
            for post in sorted(g.post_tp.unique()):
                m = g[(g.stat == "med") & (g.post_tp == post)]
                if not len(m):
                    continue
                m = m.iloc[0]
                q = g[(g.stat == "q90") & (g.post_tp == post)]
                gg = g[(g.stat == "gfp") & (g.post_tp == post)]
                c = g[(g.stat == "count") & (g.post_tp == post)]
                fold = (f"{m.fold:.2f} (2 wells)" if m.few_wells else f"{m.fold:.2f} [{m.fold_lo:.2f}, {m.fold_hi:.2f}]")
                cc = c.iloc[0] if len(c) else None
                pat = pattern[(pattern.exp == e) & (pattern.line == line)]
                p_rows.append({"exp": e, "line": SHORT[line], "int": f"T{int(m.pre_tp)}→T{int(post)}",
                               "sa": f"{10 ** m.delta_sa:.2f}×", "ut": f"{10 ** m.delta_ut:.2f}×", "fold": fold,
                               "p": pfmt(m.perm_p) + ("" if m.perm_exact else ""),
                               "q90": f"{q.fold.iloc[0]:.2f}" if len(q) else "–",
                               "gfp": f"{100 * (10 ** gg.delta_sa.iloc[0] - 1):+.0f}% / {100 * (10 ** gg.delta_ut.iloc[0] - 1):+.0f}%" if len(gg) else "–",
                               "cnt": ("–" if cc is None else f"{cc.count_ratio:.2f}" + (
                                   f" [{cc.count_ratio_lo:.2f}, {cc.count_ratio_hi:.2f}]" if not cc.few_wells else "")),
                               "pat": pat.pattern.iloc[0] if len(pat) and post == g.post_tp.max() else ""})
    alt = pp_all[(pp_all.stat.isin(["med", "q90"]))]
    a_rows = []
    for e in ["XDP5", "XDP11"]:
        for line in figs.LINE_ORDER:
            g = alt[(alt.exp == e) & (alt.line == line)]
            if not len(g):
                continue
            post = g.post_tp.max()
            def f(b, s):
                x = g[(g.baseline == b) & (g.stat == s) & (g.post_tp == post)]
                return f"{x.fold.iloc[0]:.2f} (T{int(x.pre_tp.iloc[0])})" if len(x) else "–"
            a_rows.append({"exp": e, "line": SHORT[line], "post": f"T{int(post)}", "m1": f("primary", "med"),
                           "m2": f("alt", "med"), "q1": f("primary", "q90"), "q2": f("alt", "q90")})
    x1_rows = [{"line": SHORT[r.line], "int": f"T{int(r.pre_tp)}→T{int(r.post_tp)}", "chg": f"{10 ** r.delta_sa:.2f}×",
                "ratio": f"{r.ratio_to_comparison_max:.1f}×", "n": int(r.comparison_n),
                "ok": R.badge("PASS" if r.ratio_to_comparison_max > 3 else "FAIL", "passes" if r.ratio_to_comparison_max > 3 else "does not pass")}
               for r in pp[(pp.exp == "XDP1") & (pp.stat == "med")].sort_values(["line", "post_tp"]).itertuples()]
    f5 = F["05_prepost"]
    per_exp = []
    for e in EXPS:
        items = [f5.get(f"F4_{e}"), F["03_qc_counts"].get(f"F3_{e}"), f5.get(f"F4r_{e}")]
        items = [R.figure(i) for i in items if i]
        if items:
            per_exp.append(f'<details><summary>{e}: distributions, cell counts, all timepoints</summary><div class="inner">{"".join(items)}</div></details>')
    x10b = pp[(pp.exp == "XDP10") & (pp.line == "XDP-33109.2B") & (pp.stat == "med")].set_index("post_tp").fold
    ev4 = f"""{R.figure(f5['F5'], 'Figure 4.1')}
<h3>4.1 S.A. wells vs untreated wells, every post-dose timepoint</h3>
{R.table(pd.DataFrame(p_rows), ["exp", "line", "int", "sa", "ut", "fold", "p", "q90", "gfp", "cnt", "pat"],
         ["Exp", "Line", "Pre → post", "Median ratio change, S.A.", "…, untreated", "S.A. beyond untreated (fold, 95% CI)",
          "p (well labels shuffled)", "Same for 90th pct", "GFP change S.A. / untreated", "Cells after S.A. ÷ untreated",
          "Shape of the shift (last timepoint)"], num_cols=["sa", "ut", "p", "q90"])}
<p>XDP10 has 2 S.A. wells per line: no bootstrap interval is given, and its p-value is exact (all {int(pp[(pp.exp == 'XDP10') & (pp.stat == 'med')].n_labellings.max())}
ways of labelling 2 of the wells as S.A. were enumerated). Other p-values come from 10,000 random relabellings.
"Shape of the shift" compares the change in the top decile with the middle deciles (S.A. minus untreated): "upper tail
rises most" is what a newly dead population looks like; "near-uniform" means the whole population moved (stressed but
living cells, or GFP loss).</p>
<h3>4.2 Baseline sensitivity</h3>
<p>XDP5's pre-dose image (T7) and XDP11's (T12) fall on plate-wide high-ratio days (Step 2.5). Using the image before
instead:</p>
{R.table(pd.DataFrame(a_rows), ["exp", "line", "post", "m1", "m2", "q1", "q2"],
         ["Exp", "Line", "Post", "Median fold (planned baseline)", "Median fold (earlier baseline)", "90th pct (planned)",
          "90th pct (earlier)"])}
<h3>4.3 XDP1 (whole plate dosed, no untreated wells)</h3>
<p>The plate-mean change after dosing is compared with every change of the same length between T1 and T10 that avoids the
T7 jump ("pre-dose comparison intervals"). The advance check required the change to exceed 3× the largest of those.</p>
{R.table(pd.DataFrame(x1_rows), ["line", "int", "chg", "ratio", "n", "ok"],
         ["Line", "Interval", "Median ratio change", "÷ largest pre-dose change", "Pre-dose intervals compared", "3× check"],
         raw_cols=["ok"])}
<h3>4.4 Edge column</h3>
<p>All of XDP10's S.A. wells are in column 22. XDP11 has S.A. in columns 17–22, so its edge column can be compared with
the inner S.A. columns (median ratio change T12→T13):</p>
{R.table(pd.DataFrame(edge_rows), ["line", "c22", "c1721"], ["Line", "Column 22", "Columns 17–21"])}
{R.figure(f5['F5s'], 'Figure 4.2')}
{''.join(per_exp)}"""
    out.append(step_block(
        "s4", "4", "Before vs after S.A.",
        "Did S.A. visibly change the GEDI ratio and the number of detected cells in each line and experiment, beyond what "
        "happened to untreated wells over the same interval? (Julia's request.)",
        "<p>No thresholds are used. For each well: the median (and 90th percentile) of log10 ratio, the median GFP and RFP, "
        "and the cell count, at the last pre-dose image and at each post-dose image. The change in S.A. wells minus the "
        "change in untreated wells (a difference-in-differences) removes plate-wide drift and is shown as a fold change. "
        f"Intervals: wells redrawn {int(R.step_json('05_prepost', 'manifest')['params']['B'])} times. p-values: S.A./untreated "
        "labels shuffled across wells.</p>", ev4,
        f"<p>S.A. raised the median ratio beyond the untreated change in <b>{len(rose)} of {len(ctl)}</b> cases (fold "
        f"{rng_txt(rose.fold.min(), rose.fold.max())}). XDP10 09B shows no median shift at T10 (fold {x10b.get(10, np.nan):.2f}) "
        f"but does at T9 ({x10b.get(9, np.nan):.2f}). The shape of the shift differs: upper tail rises most in "
        f"{pat_counts.get('upper tail rises most', 0)} cases, near-uniform in {pat_counts.get('near-uniform shift', 0)}, "
        f"upper tail least in {pat_counts.get('upper tail rises least', 0)} - so S.A. both creates a dead population and "
        "shifts surviving cells. Part of every rise is GFP loss in S.A. wells. Detected cells fell clearly in XDP10 and XDP11 "
        f"({rng_txt(*c1011)}), less in XDP3, and not in XDP5 (unexplained).</p>",
        decision_refs(["D04"], decs),
        "<p>No S.A. effect beyond untreated drift, or an effect before the dose. The first did not happen except XDP10 09B at "
        "T10; XDP1 cannot be judged against untreated wells and does not pass its own check. XDP5's planned baseline is an "
        "artefact day, which mostly affects the 90th percentile (4.2).</p>"))

    # ---------------- Step 5 ---------------------------------------------------------------------------
    mc_rows = [{"exp": r.exp, "line": SHORT[r.line], "n": f"{int(r.n_cells):,}",
                "k2": "yes" if r.k2_lower_is_live else "no",
                "low": f"{10 ** r.k3_lowtail_mu:.3f} ({100 * r.k3_lowtail_w:.0f}%)" if not np.isnan(r.k3_lowtail_mu) else "–",
                "live": f"{10 ** r.k3_live_mu:.3f}", "dead": (f"{10 ** r.k3_dead_mu:.3f} ({100 * r.k3_dead_w:.0f}%)"
                                                             if not np.isnan(r.k3_dead_mu) else "none ≥ 5%")}
               for r in model.itertuples()]
    f5b, f5c = F["05b_model_check"], F.get("05c_cutoff_preview", {})
    mfig = "".join(f'<details><summary>{e}: 2- vs 3-component fits</summary><div class="inner">{R.figure(f5b[f"Fmodel_{e}"])}</div></details>'
                   for e in ["XDP11", "XDP5", "XDP10", "XDP3", "XDP1"] if f"Fmodel_{e}" in f5b)
    pfig = "".join(f'<details{" open" if e == "XDP11" else ""}><summary>{e}: cutoff preview</summary><div class="inner">{R.figure(f5c[f"Fpreview_{e}"])}</div></details>'
                   for e in ["XDP11", "XDP5", "XDP10", "XDP1", "XDP3"] if f"Fpreview_{e}" in f5c)
    rmed = rules.groupby("rule")[["dead_pop_called_dead", "ut_ambiguous", "ut_death_frac", "sa_death_frac"]].median().reset_index()
    rmed["name"] = rmed.rule.map({"A": "A · D02 as written (live side from S.A. wells)", "B": "B · D02 with the live side from untreated wells",
                                  "C": "C · single cutoff: S.A. dead mean − 2.5 SD", "D": "D · equal-density boundary"})
    for c in ["dead_pop_called_dead", "ut_ambiguous", "ut_death_frac", "sa_death_frac"]:
        rmed[c] = rmed[c].map(lambda v: f"{100 * v:.0f}%")
    case_rows = []
    for r in pop.itertuples():
        b = rB[(rB.exp == r.exp) & (rB.line == r.line)]
        c = rC[(rC.exp == r.exp) & (rC.line == r.line)]
        case_rows.append({"exp": r.exp, "line": SHORT[r.line], "mode": f"{r.ut_mode:.3f}", "sd": f"{r.ut_sd_inner:.2f} / {r.ut_sd_half:.2f}",
                          "dead": (f"{r.dead_mu:.2f} (SD {r.dead_sd:.2f} log)" if r.separated else
                                   ("not separated from live" if r.dead_found else "none")),
                          "B": f"{b.live_thr.iloc[0]:.3f}–{b.dead_thr.iloc[0]:.3f} ({100 * b.dead_pop_called_dead.iloc[0]:.0f}%)" if len(b) else "–",
                          "C": f"{c.live_thr.iloc[0]:.3f} ({100 * c.dead_pop_called_dead.iloc[0]:.0f}%; untreated {100 * c.ut_death_frac.iloc[0]:.0f}%)" if len(c) else "–"})
    rules_txt = f"""<ul class="summary-list">
<li><b>Scale.</b> log10(ratio); an SD on this scale is a fold change (0.1 = ×1.26).</li>
<li><b>Cutoffs (D02).</b> Dead cutoff = live mean + {k_sd} SD; live cutoff = dead mean − {k_sd} SD. If populations were bell-shaped,
about 0.6% of each would fall past the other's cutoff. Cells between the cutoffs are dropped, as MRID does now.</li>
<li><b>Populations.</b> Found with a mixture model fitted to post-dose S.A.-well cells, per line and experiment.</li>
<li><b>Usability.</b> ≥ {params['usability']['min_cells']} cells from ≥ {params['usability']['min_wells']} wells; clear separation
(Ashman's D ≥ {params['usability']['min_ashman_d']} between the untreated live population and the S.A. dead population); the dead
component's share of S.A. cells at least {int(100 * params['usability']['min_sa_dead_excess'])} percentage points above its share of
the reference cells (untreated wells at the same timepoint; for XDP1, the same wells at T10), with the components held fixed.
A case that fails is transferred (D03), never dropped.</li>
<li><b>Transfer.</b> Cutoffs expressed in live-SD units from the line's own untreated live population in experiments with a usable
dead population, applied to the target's own untreated live population; validated by leaving one experiment out.</li>
<li><b>Checks with limits set in advance.</b> S.A. survivors vs untreated live cells; drift of the live population over time;
edge column; 2.0 / 3.0 SD; single cutoff; linear scale; fit timepoints.</li></ul>"""
    ev5 = f"""<h3>5.0 The rules as fixed on 2026-10-13</h3>{rules_txt}
<h3>5.1 Finding the dead population: two components are not enough</h3>
<p class="meta">XDP1 rows in this table use T11–T14 as originally planned; 5.2 and the Status table use T13–T14 (D14).</p>
<p>Fitting only the mixture (no cutoffs): with two components, the lower one is not the live population in
{len(model) - int(model.k2_lower_is_live.sum())} of {len(model)} cases. With three components, a separate high-ratio
dead population appears in {int(model.k3_dead_mu.notna().sum())} cases, centred at ratio
{rng_txt(10 ** model.k3_dead_mu.min(), 10 ** model.k3_dead_mu.max())}. The extra component is <i>not</i> a separate
dim-RFP artefact as first thought: its centre sits at ratio {rng_txt(10 ** model.k3_lowtail_mu.min(), 10 ** model.k3_lowtail_mu.max(), 3)}
- inside the live range, often near the untreated peak - and only {rng_txt(100 * min(lt002.values()), 100 * max(lt002.values()), 1)}%
of post-dose S.A. cells have a ratio below 0.02. It is consistent with a spread of living cells (partly S.A.-shifted, partly not) and is not interpreted further.
So only the dead component of the S.A. fit is used (D18); the live population is taken from untreated wells (D20).</p>
{R.table(pd.DataFrame(mc_rows), ["exp", "line", "n", "k2", "low", "live", "dead"],
         ["Exp", "Line", "Post-dose S.A. cells", "2-comp: lower = live?", "3-comp: lowest (share)", "3-comp: middle",
          "3-comp: dead (share)"], num_cols=["n"])}
{mfig}
<h3>5.2 Cutoff preview: what each rule would do</h3>
<p>Untreated live population: centre = peak of the untreated distribution; spread measured below the peak, where dead
cells cannot be. Its SD is {rng_txt(*sp.get('ut_sd_inner_range', [np.nan, np.nan]))} log units (inner estimate;
{rng_txt(*sp.get('ut_sd_half_range', [np.nan, np.nan]))} using the whole lower half). The S.A. dead population is narrow:
centre {rng_txt(*sp.get('dead_mu_range', [np.nan, np.nan]))}, SD {rng_txt(*sp.get('dead_sd_range', [np.nan, np.nan]))}. The two
centres are {rng_txt(*sp.get('sep_dex_range', [np.nan, np.nan]))} log units apart, but 2.5 SD on both sides needs
{rng_txt(*sp.get('needed_dex_range', [np.nan, np.nan]))}: the bands overlap in {overlap} of {len(found)} cases. Under rule B
(live side from untreated wells) the dead cutoff lies above the centre of the S.A.-dead population in {nB_above} of
{len(rB)} cases.</p>
{R.table(rmed, ["name", "dead_pop_called_dead", "ut_ambiguous", "ut_death_frac", "sa_death_frac"],
         [f"Rule (medians over {len(found)} cases)", "S.A.-dead population called dead (model; ≈99% for C by construction)", "Untreated cells dropped as ambiguous",
          "Untreated death fraction", "S.A. death fraction"], num_cols=["dead_pop_called_dead", "ut_ambiguous", "ut_death_frac", "sa_death_frac"])}
{R.table(pd.DataFrame(case_rows), ["exp", "line", "mode", "sd", "dead", "B", "C"],
         ["Exp", "Line", "Reference peak (ratio; XDP1 = own T10)", "Live SD inner / half (log)", "S.A. dead centre", "Rule B cutoffs (dead pop. called dead)",
          "Rule C cutoff (dead pop. called dead; reference death)"])}
{pfig}
<h3>5.3 What each rule does over time in untreated wells</h3>
<p>MRID applies one cutoff set per line and experiment to every timepoint. The untreated live population does not stay
put: its peak moves from ratio {rng_txt(*dsum.get('mode_T1', [np.nan, np.nan]), 3)} at T1 to {rng_txt(*dsum.get('mode_fit', [np.nan, np.nan]), 3)}
at the S.A. readout, and its spread below the peak grows from {rng_txt(*dsum.get('sd_below_T1', [np.nan, np.nan]))} to
{rng_txt(*dsum.get('sd_below_fit', [np.nan, np.nan]))} log units (above the peak at T1: {rng_txt(*dsum.get('sd_above_T1', [np.nan, np.nan]))}).
Applied to untreated wells only (XDP1: all wells up to T10):</p>
{R.table(pd.DataFrame([{"rule": k, "t1": f"{100 * v[0]:.1f}–{100 * v[1]:.1f}%",
                        "last": f"{100 * dsum['death_last_untreated_tp'][k][0]:.1f}–{100 * dsum['death_last_untreated_tp'][k][1]:.1f}%"}
                       for k, v in dsum.get('death_T1', {}).items()]),
         ["rule", "t1", "last"], ["Rule", "Untreated called dead at T1", "… at the last untreated timepoint"])}
{R.figure(F['05d_drift_preview']['Fdrift'], 'Figure 5.3') if 'Fdrift' in F.get('05d_drift_preview', {}) else ''}
<p>At T1 every rule calls few untreated cells dead (the September rule fixed this at 9.1%). Late in a run the rules
diverge, most in XDP11, where the live population drifts upward the most: rule C counts the living cells that drift past
its cutoff as dead. Whether that late drift is death or a shift of living cells is a biological question, and it decides
between the rules.</p>
<div class="decide"><h4>Recommendation for D20 (for discussion with Julia)</h4><p><b>Rule D</b> (equal-density boundary
between the untreated live population and the S.A.-dead population) as the primary cutoff, with <b>rule C</b> as the
sensitivity analysis. Both are set by the S.A.-killed cells and give a single cutoff, so no cells are dropped; D is less
sensitive to the late upward drift of living cells. C is the direct form of Julia's 2.5-SD rule (applied to the dead
population) and should be primary if the late drift is judged to be real death. Rule B (2.5 SD on both sides) misses a
large part of the S.A.-dead population. Whichever is chosen, Step 7 checks the drift per line before the cutoffs are used,
and the line-to-line spread of the cutoffs is compared with their bootstrap uncertainty.</p></div>"""
    out.append(step_block(
        "s5", "5", "Rules, and what the data forced",
        "What exactly will be computed, and do the planned rules work on the real distributions?",
        "<p>The rules below were fixed by Austin on 2026-10-13, before any threshold was fitted (D08). Then the mixture alone "
        "was fitted (5.1), and the candidate cutoff rules were previewed on every case (5.2), to check them against the real "
        "distributions before computing the final thresholds.</p>", ev5,
        "<p>The S.A.-dead population is clear and consistent where it exists. The live population is too broad on the log "
        "scale for 2.5 SD to be applied on both sides, so the cutoff rule needs a decision (D20). Cases without a usable "
        "dead population (all of XDP3, several others; Status table) will be transferred.</p>",
        decision_refs(["D02", "D08", "D17", "D18", "D20"], decs),
        "<p>A dead population at different ratios in different experiments, or none at all. Neither: where S.A. produced one, "
        "it sits at a similar ratio everywhere. What did fail was the plan to take both cutoffs from the same SD rule.</p>"))

    out.append(steps_6_to_10(decs, params, F, S))
    out.append(steps_11_to_13(decs, F, S))
    out.append(steps_14_to_17(decs, F, S))

    # ---------------- decision log -------------------------------------------------------------------------
    drows = []
    for d in decs:
        opts = ""
        if d.get("options"):
            opts = "<ul>" + "".join(f"<li><b>{E(str(o.get('label')))}</b>" + (f" - for: {E(str(o.get('pros')))}" if o.get("pros") else "")
                                    + (f"; against: {E(str(o.get('cons')))}" if o.get("cons") else "") + "</li>" for o in d["options"]) + "</ul>"
        who = f" · decides: {E(str(d['decider']))}" if d.get("decider") else ""
        drows.append(f"""<div class="box" id="{d['id']}"><span class="k">{d['id']} · {E(str(d.get('step', '')))} · {E(str(d.get('date', '')))}{who}</span>
<p><b>{E(str(d['question']))}</b></p>{opts}<p>{E(str(d['decision']))}</p>{('<p class="meta">' + E(str(d['rationale'])) + '</p>') if d.get('rationale') else ''}
<p>{R.badge(d['status'])} {E(str(d.get('approved_by', '')))}</p></div>""")
    out.append(f'<section class="step" id="decisions"><h2>Decision log</h2>{"".join(drows)}</section>')

    # ---------------- checks ---------------------------------------------------------------------------
    cks = checks.copy()
    cks["kind"] = np.where(cks.step.isin(input_steps), "input", "rule / method")
    cks["status_b"] = cks.status.map(lambda s: R.badge(s, s))
    cks["value_s"] = cks["value"].map(lambda v: E(json.dumps(v, default=str))[:220])
    fails = cks[cks.status == "FAIL"].groupby("step").size().to_dict()
    out.append(f"""<section class="step" id="checks"><h2>All checks</h2>
<p>{len(cks)} checks. Input checks: {int((ic.status == 'PASS').sum())} pass, {int((ic.status == 'WARN').sum())} informational,
{n_in_fail} fail. Rule and method checks: {int((rc.status == 'PASS').sum())} pass, {int((rc.status == 'WARN').sum())}
informational, {int((rc.status == 'FAIL').sum())} fail ({', '.join(f'{k}: {v}' for k, v in fails.items())}) - planned rules
or method validations that did not hold, discussed in Steps 3–5.</p>
{R.table(cks, ["step", "kind", "id", "status_b", "description", "value_s"], ["Step", "Kind", "Check", "Status", "What", "Value"],
         raw_cols=["status_b", "value_s"])}</section>""")

    gl = [("GEDI ratio", "Per cell: mean RFP ÷ mean GFP intensity. Low = alive, high = dead."),
          ("log scale", "We work with log10(ratio). An SD on this scale is a fold change: 0.1 = ×1.26, 0.3 = ×2."),
          ("live / dead cutoff", "Below the live cutoff a cell is called live, above the dead cutoff dead, in between it is dropped (MRID)."),
          ("S.A.", "Sodium arsenite, added on purpose to kill cells and show where dead cells sit."),
          ("untreated wells", "Same line, same plate, no drug and no siRNA."),
          ("difference-in-differences", "Change in S.A. wells minus change in untreated wells over the same interval."),
          ("pre-dose comparison intervals", "XDP1 only: changes of the same length between T1 and T10, before any S.A."),
          ("mixture model", "A fit of the ratio distribution as a sum of bell curves; one of them is the S.A.-dead population."),
          ("Ashman's D", "Distance between two population centres in units of their spread; above 2 = clearly separate."),
          ("bootstrap", "Re-computing a result on wells drawn with replacement, to get an interval."),
          ("onset scan", "Trying each timepoint as the start of the S.A. effect and keeping the one that fits best."),
          ("transfer", "Setting cutoffs where S.A. gave no usable dead population, from offsets measured in other experiments."),
          ("review points", "A = data, timing, before/after and rules; B = the thresholds; C = final sign-off; D = counts and dropout (Steps 14-17)."),
          ("capture fraction (c)", "Of the live cells S.A. removed, the share still seen as GEDI-dead; below 1 = the rest dropped out (Step 15)."),
          ("visibility window (V)", "How many hours a dead cell stays detectable as GEDI-dead; GFP dropout shortens it."),
          ("live-cell retention", "How many of its live cells a line keeps over time, relative to another line on the same plate (Step 16)."),
          ("S.A.-corrected odds ratio", "MRID's odds ratio × c(reference line) ÷ c(comparison line) (Step 17)."),
          ("validity gate (D27)", "A capture fraction is used only if the S.A. wells also lost detected cells (95% upper bound of detected ÷ expected below 1) and its bootstrap stays above 0."),
          ("k0", "Start of the counting window: the early timepoint where detected cells per untreated well peak, after GFP maturation."),
          ("control-to-line retention ratio per week", "Share of live cells the control keeps in a week ÷ the share the line keeps; above 1 = the line keeps fewer (Step 16)."),
          ("calibration test", "Asks whether a line's visible GEDI death accounts for its loss of live cells (Step 16.2)."),
          ("Hartung-Knapp interval", "A random-effects interval that stays honest with few experiments (3-6 here); wider than the usual one."),
          ("range for a new experiment", "Prediction interval: where the result of one more experiment would be expected to fall."),
          ("placebo comparison", "The S.A. estimator applied before any dose; shows how far wells drift apart by chance.")]
    out.append('<section class="step" id="glossary"><h2>Glossary</h2><div class="table-wrap"><table><tbody>'
               + "".join(f"<tr><td><b>{E(a)}</b></td><td>{E(b)}</td></tr>" for a, b in gl) + "</tbody></table></div></section>")

    g = git_state()
    inp = "".join(f"<tr><td class='mono'>{E(k)}</td><td class='mono'>{E(v['source'])}</td><td class='num'>{v['bytes']:,}</td>"
                  f"<td class='mono'>{v['sha256'][:16]}…</td></tr>" for k, v in staging.items() if v["bytes"] > 1_000_000)
    cfg_sha = {n: sha256_file(CONFIG_DIR / n)[:16] for n in ("experiments.yaml", "params.yaml", "decisions.yaml")}
    out.append(f"""<section class="step" id="provenance"><h2>Provenance</h2>
<p class="meta">Code: <code>analyses/xdp_rgedi_rework/</code> in nextflow-cluster, branch <code>{E(g['branch'])}</code>, HEAD
<code>{E(g['head'][:10])}</code>, package content hash <code>{g['package_content_sha256'][:12]}</code>
({'uncommitted changes' if g['package_dirty'] else 'clean'}). Config hashes: {', '.join(f'{k} {v}' for k, v in cfg_sha.items())}.
Python {E(versions()['python'])}, pandas {E(versions()['pandas'])}, scikit-learn {E(versions()['sklearn'])}, statsmodels
{E(versions()['statsmodels'])}; R 4.5.1 for MRID's ratio code. Outputs: <code>/Volumes/Finkbeiner-Kaye/XDP/RGEDI-full-rework/</code>.</p>
<div class="table-wrap"><table><thead><tr><th>Input</th><th>Source on the share</th><th class="num">Bytes</th><th>SHA-256</th></tr></thead>
<tbody>{inp}</tbody></table></div></section>""")
    out.append('<footer><span>XDP RGEDI threshold evidence · ' + E(label) + '</span></footer></main></div>')
    return "\n".join(out)


def main() -> None:
    label = sys.argv[1] if len(sys.argv) > 1 else "draft"
    body = build(label)
    from lib.paths import OUT_ROOT
    stamp = time.strftime("%Y%m%d")
    name = f"RGEDI_SA_thresholds_evidence_{stamp}_{label}.html"
    local = LOCAL_OUT / "docs"
    (local / "artifact").mkdir(parents=True, exist_ok=True)
    (local / "artifact" / name).write_text(body)
    head_end = body.index("</style>") + len("</style>")
    full = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, '
            'initial-scale=1, viewport-fit=cover">' + body[:head_end] + "</head><body>" + body[head_end:] + "</body></html>")
    (local / name).write_text(full)
    mb = len(full.encode()) / 1e6
    print(f"wrote {local / name} ({mb:.2f} MB)")


if __name__ == "__main__":
    main()
