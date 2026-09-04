#!/usr/bin/env python
"""Build the timepoint -> elapsed-hours table from ACTUAL raw-image acquisition times.

RGEDI imaging schedules are nominally fixed (e.g. 24 h), but a paused/restarted run
drifts off-sync, so the survival model's time axis must use the real fire times, not
the scheduled interval. Each raw IXM/MetaSeries TIFF carries a `DateTime` tag
(`YYYYMMDD HH:MM:SS.sss`); for each timepoint we take the EARLIEST tile of a reference
well (the first image the scope fired that run) and report hours since T1.

Usage: python timepoints_from_images.py --raw-dir <EXP-RGEDI dir> --well D03 \
           --timepoints 1-9 --channel FITC --out actual_hours.csv
Output: CSV `Timepoint,Hour`.
"""
import argparse
import datetime
import glob
import os
import tifffile


def read_dt(f):
    with tifffile.TiffFile(f) as t:
        v = t.pages[0].tags['DateTime'].value
    return datetime.datetime.strptime(str(v).strip(), "%Y%m%d %H:%M:%S.%f")


def parse_tps(spec):
    out = []
    for part in str(spec).split(','):
        if '-' in part:
            a, b = part.split('-'); out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', required=True, help='<EXP>-RGEDI raw dir holding per-well tile folders')
    ap.add_argument('--well', required=True, help='reference well, e.g. D03')
    ap.add_argument('--timepoints', required=True, help='e.g. 1-9 or 1,2,3')
    ap.add_argument('--channel', default='FITC')
    ap.add_argument('--out', default='actual_hours.csv')
    a = ap.parse_args()

    rows, t0 = [], None
    for t in parse_tps(a.timepoints):
        files = sorted(glob.glob(f"{a.raw_dir}/{a.well}/*_T{t}_0-1_{a.well}_*_{a.channel}_*.tif"))
        if not files:
            print(f"  T{t}: no {a.channel} tiles found -- skipped", flush=True); continue
        dts = []
        for f in files:
            try:
                dts.append(read_dt(f))
            except Exception:
                pass
        if not dts:
            print(f"  T{t}: no readable DateTime -- skipped", flush=True); continue
        d = min(dts)                                   # first image fired this timepoint
        t0 = t0 or d
        hour = round((d - t0).total_seconds() / 3600.0, 3)
        rows.append((t, hour))
        print(f"  T{t}: {d}  elapsed={hour:.2f} h", flush=True)

    with open(a.out, 'w') as fh:
        fh.write("Timepoint,Hour\n")
        for t, h in rows:
            fh.write(f"{t},{h}\n")
    print(f"wrote {a.out}: {len(rows)} timepoints (actual acquisition times)", flush=True)


if __name__ == '__main__':
    main()
