# Ultrack tuning / QC (hevo-pmsG-1)

Evaluation that set the tuned defaults in `bin/tracking_ultrack.py`. All read-only,
run in the dedicated `ultrack.sif` (sweep) or the main SIF (compare/overlay).

| script | what it does |
|---|---|
| `ultrack_sweep.py` | F6 sweep of `max_distance {300,450,600}` x `appear/disappear {-0.001,-1.0}`; reports track length / T0->end survival vs the proximity baseline (labels_to_contours computed once, reused). |
| `track_compare.py` | Ultrack vs proximity on F6: track-length distribution + persistence + dropout. |
| `overlay_tracks.py` | Draws track trajectories on the F6 montage, proximity vs Ultrack, for tracks >=10 frames. |

## Result (F6, 2026-09-08)

| config | tracks | @T0 | median len | T0->end survival |
|---|---|---|---|---|
| Proximity (md450+motion) | 383 | 262 | 13 | 31% (dropout 69%) |
| Ultrack md=450 appear/disappear=-1.0 | 128 | 124 | 19 | **73%** |
| Ultrack md=600 appear/disappear=-1.0 | 131 | 123 | 19 | 75% |

**The appear/disappear penalty was the key lever** (weak default -0.001 fragmented every
max_distance to ~2-4% survival). Chose **md=450** over 600: ~same completeness, tighter radius
trims the ~2% of >400px jumps (candidate ID-swaps). Ultrack median step 30px (tighter than
proximity 36px) -> the completeness gain is real, not from spurious long links. Gurobi solved
on the compute node (WLS license works there). min_area lowered 500->250 to recover small cells
(tuned Ultrack saw ~123 T0 cells vs proximity ~262). See [[gedi-canonical-v3-approach]].
