#!/usr/bin/env python
"""Generate an MRID plate layout (Sci_WellID, Sci_SampleID, Drug) for the ratio stage.

MRID needs well -> sample(cell line) -> drug. We already have per-experiment platemaps
(e.g. XDP<N>-RGEDI-platemap.csv: well, celltype, condition, name, dosage, kind). This
maps such a source into MRID's 3-column layout so users never hand-make it. Column names
are configurable to accommodate different platemap formats.
"""
import argparse
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('source', help='source platemap csv')
    ap.add_argument('--out', default='plate_layout.csv')
    ap.add_argument('--well-col', default='well')
    ap.add_argument('--sample-col', default='celltype', help='cell line / sample id column')
    ap.add_argument('--drug-col', default='', help='drug column (blank -> "No drug")')
    a = ap.parse_args()
    df = pd.read_csv(a.source)
    df.columns = [c.strip().lstrip('﻿') for c in df.columns]
    out = pd.DataFrame({
        'Sci_WellID': df[a.well_col].astype(str),
        'Sci_SampleID': df[a.sample_col].astype(str),
        'Drug': (df[a.drug_col].astype(str) if a.drug_col and a.drug_col in df.columns else 'No drug'),
    })
    out = out[out.Sci_WellID.notna() & (out.Sci_WellID != 'nan')].drop_duplicates('Sci_WellID')
    out.to_csv(a.out, index=False)
    print(f"wrote {a.out}: {len(out)} wells, {out.Sci_SampleID.nunique()} samples, "
          f"{out.Drug.nunique()} drug(s)")


if __name__ == '__main__':
    main()
