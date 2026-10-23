# GEDI analysis rules — CANONICAL v3 (Hal3, Hal4, all future plates)

## 1. Measurement
Consume the pre-computed **MASKTRACKED** GFP-DMD1 label images (segmentation + tracking are already done).
Per (well, tile, timepoint, track_id): largest connected component of that label → `area`,
`rfp_mean` (RFP1), `gfp_mean` (GFP-DMD1). Track identity = the label integer.

## 2. GEDI2 — two-pass, per-timepoint background
```
GEDIratio2 = (RFP_cell − median RFP of that tile AT THAT TIMEPOINT) / GFP_cell
```
- Background is **PER TIMEPOINT**: `groupby(["well","tile","timepoint"])`. Never pool across the timecourse.
- **Two-pass:** pass 1 computes provisional deaths; pass 2 recomputes the tile median with
  **all post-death observations removed**, then re-derives deaths. Dead cells are RFP-bright and
  must not inflate the background.

## 3. Death call
- Threshold **GEDIratio2 > 0.025**.
- **Death = above threshold on ONE OR MORE frames** (`n_above >= 1`). Frames need not be consecutive.
- **Death time = the FIRST frame above threshold.**

## 4. Tracking / survival rules
- **Present at T0** — track must start at the first frame.
- **Alive at T0** — GEDIratio2 ≤ 0.025 at that first frame; otherwise drop the cell.
- **Once dead, stays dead** — no observations counted after the death call, anywhere
  (survival, background, plots).
- **KM CENSORING — never drop data.** A track lost while still alive is **censored** at its last
  observed frame (`event=0`): it contributes follow-up time up to that point and is then removed
  from the at-risk set. A lost track is **NOT** a death. Do not treat loss of GFP/track as death.
  Do not exclude these cells from the cohort.
- Completeness is judged against **each well's own final frame** (wells can end on different frames).

## 5. Metric
**GEDI+ death only**, Kaplan–Meier. No "total cell loss"/attrition metric.

## 6. Stimulation map — verify empirically, per well
Read the **BLUE-DMD-blocked** channel (saturated ≈65535 where the DMD fired, ~360 where it did not).
Tier tiles: tier0={1,6,11,16}, tier1={2,7,12,13}, tier2={3,8,9,14}, tier3={4,5,10,15}.
- Standard wells: `[0,0,10000,10000]` → stim tiles {3,4,5,8,9,10,14,15}.
- Hal3 titration wells: G8/G10 `[0,500,5000,10000]`, G9/G11 `[0,1000,2500,7500]`.
- **Hal4 G8–G11 were programmed with the Hal3 titration pattern** (G8/G10 `[0,500,5000,10000]`,
  G9/G11 `[0,1000,2500,7500]`) — they are a valid dose-response, not faulty wells.
- **Hal4 B5 = no tiles stimulated** → all 16 tiles are 0 ms. **B6 = all tiles stimulated** → all 16
  tiles are 10,000 ms. Keep both; they also serve as cross-tile bleedthrough controls.

## 7. Timing
Frame→time from **actual image file timestamps**, never the run-sheet's planned times (those are
fictional). Per-plate `frame_time.csv`. Cadence is irregular. Hal3 ≈118 h; Hal4 ≈137 h.

## 8. Stats
KM cumulative death; Greenwood 95% CI; log-rank vs DMSO; Benjamini–Hochberg FDR per comparison
family. Note the pseudoreplication caveat (cells within a well are correlated).

## 9. Process
Do not re-run a finished analysis unless asked. Work only on the plate the user asked about.
Verify the stimulus was actually delivered before drawing any biological conclusion.
