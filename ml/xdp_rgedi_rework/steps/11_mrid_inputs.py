"""Step 11: MRID input trees (ratio_output + timepoints) for every threshold set (decisions D20-D23).

Sets (cutoffs per line x experiment; direct where usable, transferred otherwise):
  D_sd        rule D, live-SD transfer                       (track 1, primary)
  D_abs       rule D, absolute-offset transfer               (D21 sensitivity)
  C_sd        rule C, live-SD transfer                       (rule C sensitivity)
  C_abs       rule C, absolute-offset transfer
  B           rule B (two cutoffs; cells between dropped)   (track 2)
  D_sd_drift  D_sd with the cutoff moved with the untreated live peak at each timepoint, for the cases
              whose live population drifts (MODERATE or MAJOR in step 07)   (D22 sensitivity)
Cells used (D23): in-scope lines only; no siRNA wells; no S.A. wells; XDP1 (whole plate dosed) up to T10.
Classification follows MRID's mrid_build_combined_livedead exactly: live if ratio < live cutoff, dead if
ratio > dead cutoff, otherwise dropped; classifier.score.dead = ratio > live cutoff.
Layout: <out>/input/<set>/<EXPT>-RGEDI/<EXPT>-RGEDI_ratio_output.csv and <EXPT>-RGEDI_timepoints.csv.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib import cfg, data, figs, livedead  # noqa: E402
from lib.paths import LOCAL_OUT  # noqa: E402
from lib.provenance import StepRecorder  # noqa: E402

STEP = "11_mrid_inputs"
SETS = livedead.SETS


def cells_for_mrid(exp: str) -> pd.DataFrame:
    d = data.load(exp, columns=["well", "line", "condition", "tp", "obj", "ratio", "in_scope"])
    if exp == "XDP1":
        d = d[d.tp <= 10]
    else:
        d = d[d.condition == "untreated"]
    return d


def main() -> None:
    rec = StepRecorder(STEP, LOCAL_OUT / STEP)
    ctx = livedead.drift_context()
    summary = {"drifting_cases": livedead.drifting_cases(ctx), "sets": {}}
    cells = {exp: cells_for_mrid(exp) for exp in figs.EXP_ORDER}
    for name, (_, _, form) in SETS.items():
        cut = livedead.cutoff_table(form)
        rows = []
        for exp in figs.EXP_ORDER:
            e = cfg.experiments()[exp]
            d = livedead.classify(cells[exp], exp, name, cut, ctx)
            d = d[d.live_guesses.notna()]
            out = pd.DataFrame({
                "filenames": exp + "_T" + d.tp.astype(str) + "_" + d.well + "_" + d.obj.astype(str),
                "Sci_WellID": d.well, "Sci_SampleID": d.line, "Drug": "", "Timepoint": d.tp, "ratio": d.ratio,
                "live_guesses": d.live_guesses, "classifier.score.live": np.where(d.ratio > d.lo, 0.0, 1.0),
                "classifier.score.dead": np.where(d.ratio > d.lo, 1.0, 0.0)})
            folder = rec.out_dir / "input" / name / e["expt_name"]
            folder.mkdir(parents=True, exist_ok=True)
            out.reset_index(drop=True).to_csv(folder / f"{e['expt_name']}_ratio_output.csv")
            tps = cfg.read_timepoints(exp)
            tps = tps[tps.tp.isin(out.Timepoint.unique())]
            pd.DataFrame({"TimePoint": "T" + tps.tp.astype(str), "ElapsedHours": tps.hours}).to_csv(
                folder / f"{e['expt_name']}_timepoints.csv", index=False)
            g = out.groupby(["Sci_SampleID", "Timepoint"]).agg(live=("live_guesses", "sum"), cells=("live_guesses", "size"))
            g["dead"] = g.cells - g.live
            g = g.reset_index().assign(exp=exp, set=name)
            rows.append(g)
        cnt = pd.concat(rows)
        n_in = sum(len(c) for c in cells.values())
        rec.check(f"{name}.no_duplication", f"{name}: classified + dropped cells never exceed the cells available",
                  int(cnt.cells.sum()) <= n_in, {"classified": int(cnt.cells.sum()), "available": n_in})
        cnt.to_csv(rec.out_dir / f"counts_{name}.csv", index=False)
        summary["sets"][name] = {"cells_classified": int(cnt.cells.sum()), "dead": int(cnt.dead.sum()),
                                 "death_frac": float(cnt.dead.sum() / cnt.cells.sum())}
        print(name, summary["sets"][name], flush=True)
    rec.summary = summary
    rec.write()


if __name__ == "__main__":
    main()
