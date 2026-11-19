#!/usr/bin/env python3
"""Compare two montage_percell.py runs (e.g. a test run against expected output).

Reads ``perwell.csv`` and ``percell.csv`` from two run folders and reports, for
each well, the difference in every numeric column. Uses only the standard
library, so it runs with any Python 3 (inside or outside the container).

Exit code 0 = PASS, 1 = FAIL.

Example:
    python compare_runs.py --expected <kit>/expected --got <out>/latest --exact
    python compare_runs.py --expected <kit>/expected --got <out>/latest --rtol 0.05
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

KEY = ("experiment", "well", "timepoint")
TEXT_COLS = {"experiment", "well", "timepoint", "cell_line", "genotype"}


def read_rows(path: str) -> List[Dict[str, str]]:
    """Read a CSV file into a list of dicts.

    Args:
        path: CSV file path.

    Returns:
        One dict per row.
    """
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def rel_diff(a: float, b: float) -> float:
    """Relative difference of b from a, with a floor so zeros do not divide.

    Args:
        a: Expected value.
        b: Observed value.

    Returns:
        abs(b - a) / max(abs(a), 1.0).
    """
    return abs(b - a) / max(abs(a), 1.0)


def compare_perwell(exp_rows: List[Dict[str, str]], got_rows: List[Dict[str, str]],
                    rtol: float) -> Tuple[bool, List[str]]:
    """Compare per-well tables column by column.

    Args:
        exp_rows: Expected perwell rows.
        got_rows: Observed perwell rows.
        rtol: Allowed relative difference (0 = values must be equal).

    Returns:
        (passed, report lines).
    """
    lines, ok = [], True
    got = {tuple(r.get(k, "") for k in KEY): r for r in got_rows}
    for e in exp_rows:
        key = tuple(e.get(k, "") for k in KEY)
        g = got.get(key)
        name = "/".join(key)
        if g is None:
            lines.append(f"  {name}: MISSING in the new run")
            ok = False
            continue
        worst = []
        for col, ev in e.items():
            if col in TEXT_COLS:
                if g.get(col) != ev:
                    worst.append(f"{col} {ev!r} -> {g.get(col)!r}")
                continue
            try:
                a, b = float(ev), float(g.get(col, "nan"))
            except ValueError:
                worst.append(f"{col}: not a number")
                continue
            if rel_diff(a, b) > rtol:
                worst.append(f"{col} {a:g} -> {b:g} ({100 * rel_diff(a, b):.1f}%)")
        if worst:
            ok = False
            lines.append(f"  {name}: DIFFERENT: " + "; ".join(worst))
        else:
            lines.append(f"  {name}: n_somas={g['n_somas']} "
                         f"with_neurite={g['n_somas_with_neurite']} "
                         f"mean_len_per_soma={g['mean_len_per_soma']}  OK")
    return ok, lines


def compare_percell(exp_rows: List[Dict[str, str]], got_rows: List[Dict[str, str]],
                    exact: bool) -> Tuple[bool, List[str]]:
    """Compare per-cell tables: row counts per well, and every row if exact.

    Args:
        exp_rows: Expected percell rows.
        got_rows: Observed percell rows.
        exact: If True, every row must be identical.

    Returns:
        (passed, report lines).
    """
    def by_well(rows: List[Dict[str, str]]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
        out: Dict[Tuple[str, ...], List[Dict[str, str]]] = {}
        for r in rows:
            out.setdefault(tuple(r.get(k, "") for k in KEY), []).append(r)
        return out

    lines, ok = [], True
    e_w, g_w = by_well(exp_rows), by_well(got_rows)
    for key, erows in e_w.items():
        grows = g_w.get(key, [])
        name = "/".join(key)
        msg = f"  {name}: {len(erows)} cells expected, {len(grows)} in the new run"
        if exact:
            order = lambda r: int(r["soma_id"])  # noqa: E731
            same = sorted(erows, key=order) == sorted(grows, key=order)
            msg += "; all rows identical" if same else "; ROWS DIFFER"
            ok = ok and same
        elif len(erows) and abs(len(grows) - len(erows)) / len(erows) > 0.05:
            msg += "; COUNT DIFFERS by more than 5%"
            ok = False
        lines.append(msg)
    return ok, lines


def main() -> int:
    """CLI entry point."""
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--expected", required=True, help="folder with the expected perwell.csv + percell.csv")
    ap.add_argument("--got", required=True, help="run folder to check (e.g. <out-dir>/latest)")
    ap.add_argument("--exact", action="store_true",
                    help="require identical numbers (use for the CPU-from-cache test)")
    ap.add_argument("--rtol", type=float, default=0.05,
                    help="allowed relative difference per per-well column when not --exact "
                         "(default 0.05 = 5%%)")
    args = ap.parse_args()
    rtol = 0.0 if args.exact else args.rtol

    ok = True
    print(f"Expected: {os.path.abspath(args.expected)}")
    print(f"New run : {os.path.realpath(args.got)}")
    print(f"Mode    : {'exact' if args.exact else f'tolerance {100 * rtol:.0f}%'}\n")

    print("perwell.csv")
    p_ok, lines = compare_perwell(read_rows(os.path.join(args.expected, "perwell.csv")),
                                  read_rows(os.path.join(args.got, "perwell.csv")), rtol)
    print("\n".join(lines))
    ok = ok and p_ok

    print("\npercell.csv")
    c_ok, lines = compare_percell(read_rows(os.path.join(args.expected, "percell.csv")),
                                  read_rows(os.path.join(args.got, "percell.csv")), args.exact)
    print("\n".join(lines))
    ok = ok and c_ok

    print("\nRESULT: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
