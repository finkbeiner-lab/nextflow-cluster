# Code Review — Nextflow Pipeline (core bundled-workflow files)

Scope: the 11 core Python files used by the bundled IXM/STD workflows, plus
`run.sh`, `nextflow.config`, and the container definition. Reviewed for
**efficiency, speed, reliability, and safety** — not style.

Two findings have already been fixed in this pass (see "Fixed" at the bottom).
Everything else is listed with a severity and a concrete fix, most severe
first, so you can decide what to schedule. The HIGH items that change pipeline
behavior or scientific output are deliberately **left for you to approve and
test on real data** rather than applied blind.

---

## CRITICAL / HIGH — should be fixed before the next production run

### 1. Parallel wells clobber shared CSV outputs (data loss)
`bin/tracking_montage.py:1058-1060` and `:1213-1216`
`TRACKING_MONTAGE` runs one Slurm job per well **in parallel** (`tag "$well"`
in `modules.nf`). Both `save_tracking_statistics()` (`to_csv(..., mode='w')`)
and `run()` (check-then-append) write to the **same** shared files
(`{experiment}_tracking-info.csv`, `{experiment}_tracked_montage_summary.csv`).
Concurrent wells race on the exists-check and the last well to finish
overwrites the others — silent loss of tracking results.
**Fix:** write per-well files (`{experiment}_{well}_tracking-info.csv`) and
merge in a downstream step, or use file locking (`fcntl`/`portalocker`) with
true append-only writes.

### 2. Guaranteed crash on incomplete tile sets
`bin/montage.py:378-385`
`update_field`, `experimentdata_id`, `welldata_id`, `channeldata_id`,
`well`/`timepoint` are only assigned inside nested `if` branches, but
`self.batch_updates.append(...)` references them unconditionally. Any
well/timepoint/channel group with a missing tile (common in real acquisitions)
raises `UnboundLocalError` and kills the whole montage job.
**Fix:** initialize those names to `None` up front and only append inside a
guarded branch; skip incomplete groups with a logged warning.

### 3. Per-label multiprocessing blowup in tracking
`bin/tracking_montage.py:123-129` (called from `:711`)
`args_list = [(mask, label) for label in labels]` ships the **entire** montage
mask to `Pool.map` once per cell label (hundreds–thousands of times per
timepoint), and a fresh `Pool` is created/destroyed per mask. Severe
CPU/memory/IPC overhead; can OOM compute nodes.
**Fix:** create one Pool per well (or use `Pool(initializer=...)` to set the
mask once per worker), or drop multiprocessing entirely and use
`scipy.ndimage.find_objects` / `skimage.measure.regionprops` on bounding-box
crops.

### 4. Alignment recomputes the same shift N times and clobbers its own output
`bin/align_montage_dft.py` (grouping `:149`, writes `:173`/`:199`)
Because the montage step writes the same `newimagemontage` path to every
pre-montage tile row, `align_tiles()` (grouped by `['well','tile']`) repeats
the identical DFT computation + warp + TIFF write once per tile index, all to
the same output path — up to N× wasted compute/I/O, only the last write
survives.
**Fix:** dedupe on `newimagemontage` (or group by `well`,`timepoint` only),
compute the shift once, and update all matching `tiledata_id`s.

### 5. Silently swallowed DB delete failures
`bin/segmentation_montage.py:600-616`
`bulk_delete_celldata` wraps both bulk deletes in `except Exception: pass` with
a comment claiming it logs — but there is no log call. A failed delete leaves
stale celldata rows that duplicate freshly-inserted cells, with no trace.
**Fix:** `except Exception as exc: logger.error(...)` and consider retry/raise.

### 6. DB credentials: superuser + plaintext on shared storage
`bin/sql.py:58` (connection string)
The connection uses the `postgres` superuser with a plaintext password read
from a lab-shared CSV (`/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv`).
Anyone who can read that path gets superuser on the `galaxy` DB.
**Fix:** use a least-privilege role scoped to the app's tables; move the secret
to an env var / secret manager, or at minimum lock the file down to `0400`
owned by a service account.

---

## MEDIUM — reliability & performance

- **`bin/sql.py` — connection/reflection storm.** `Database.__init__` runs 13
  `has_table()` round-trips plus **two** full `meta.reflect()` passes on every
  instantiation and uses `NullPool` (no connection reuse). `Ops` methods each
  build their own `Database()` (e.g. `get_celldata_df` builds two). With up to
  22 concurrent wells this multiplies badly. **Fix:** reuse one `Database` per
  unit of work; reflect once per process; consider a pooled engine.
- **Unbounded bulk INSERT.** `bin/sql.py:371-381` `add_row` sends an arbitrarily
  large multi-VALUES statement; large wells can hit Postgres parameter limits.
  **Fix:** chunk inserts (a few thousand rows each).
- **N+1 DB updates in alignment.** `bin/align_montage_dft.py:173,199` update one
  row at a time inside loops, unlike montage/segmentation which batch. **Fix:**
  accumulate and batch-update.
- **Missing `cv2.imread` None-checks.** `bin/utils.py:210-212,711,719`;
  `bin/align_montage_dft.py:155,192`; `bin/montage.py:281` — a corrupt/missing
  file yields `None` and crashes deep in numpy instead of naming the bad file.
  **Fix:** check `is None`, log the path, skip/raise clearly.
- **Pandas NaN mask crash.** `bin/align_montage_dft.py:132`
  `.str.endswith('.tif')` returns NaN for missing paths and blows up boolean
  indexing. **Fix:** `.str.endswith('.tif', na=False)`.
- **Unchecked ImageMagick subprocesses.** `bin/utils.py:746,770,793` never
  inspect `returncode`/stderr, so `convert` failures pass silently. **Fix:**
  `subprocess.run(..., check=True, capture_output=True)`.
- **Unchecked `cv2.imwrite`.** `bin/tracking_montage.py:1157` — OpenCV returns
  `False` on write failure instead of raising. **Fix:** check the return value.
- **Redundant work / peak memory.** `bin/tracking_montage.py:691` vs `:1127`
  reads the same mask twice; `bin/overlay_montage.py:165-176` re-pickles the
  whole multi-well summary per timepoint; `bin/montage.py:247-331` holds all
  raw tiles + the assembled montage in memory at once. **Fix:** cache/filter to
  the current well; pre-allocate the montage and write tiles into slots.
- **ThreadPoolExecutor for CPU-bound work.** `bin/segmentation_montage.py:264` —
  labeling/regionprops are largely GIL-bound; thread count may not scale.
  **Fix:** benchmark `ProcessPoolExecutor`.
- **Missing NULL check.** `bin/db_util.py:71-75` checks `imagedir` but not
  `analysisdir`; a NULL `analysisdir` gives an opaque `TypeError`. **Fix:** check
  both.
- **Cross-well data leak (latent).** `bin/db_util.py:186-188`
  `get_trackedmaskpath_from_other_channel` filters only by `tile`+`timepoint`,
  not by well/experiment; tile numbers repeat, so it can return another well's
  path. Currently unused but a landmine. **Fix:** scope by `welldata_id`.
- **Divide-by-zero on blank images.** `bin/utils.py:665-682` `zerone_normalizer`
  divides by `(max-min)`, producing `inf`/`nan` on flat images. **Fix:** guard
  `max == min`.
- **`run.sh` allocates 2 nodes for the driver.** `#SBATCH -N 2` reserves two
  whole nodes just to run the Nextflow driver, which then submits its own
  per-process Slurm jobs. **Fix:** `-N 1` with modest CPU/mem for the driver.
- **Missing `finkbeiner.config.template`.** Both `run.sh:41,92` and
  `setup-workspace.sh:26` reference `finkbeiner.config.template`, which does not
  exist in the repo — `setup-workspace.sh` fails and the run.sh error messages
  point users at a missing file. **Fix:** add the template (a copy of
  `finkbeiner.config` with placeholders).

---

## LOW — cleanup / hardening

- `bin/segmentation_montage.py:353-355` fallback allocates a full-image array
  instead of scalar `65535`.
- `bin/segmentation_montage.py:262-270` `chunk_size` computed but never used.
- `bin/montage.py:278` TOCTOU on `makedirs`; use `exist_ok=True`.
- `bin/tracking_montage.py:578-591` channel prefix substitution silently leaves
  the wrong channel if no known prefix matches — log/raise instead.
- `bin/segmentation_helper_montage.py` has four near-duplicate update functions
  differing only by name/typo — consolidate to one, alias the rest.
- `bin/utils.py:409-432` `pickle.load` from shared storage is an RCE vector if
  tampered with — prefer JSON for simple caches, restrict dir perms.
- `bin/utils.py:525-546` `find_ind_of_filename` returns `False` for "not found"
  (index 0 is falsy) — return `Optional[int]`.
- `bin/utils.py:559-563` `return_image_indices` re-globs the whole directory
  twice per image — compute the index once.
- `bin/utils.py:573` path built with `+` instead of `os.path.join`.

---

## No SQL-injection risk found
All query paths in `sql.py`/`db_util.py` use SQLAlchemy Core constructs
(`filter_by(**kwargs)`, bound params, `.values()`), so user-supplied values are
properly parameterized. `validate_config.py` is clean (local file parse, no
shell/eval/SQL).

---

## APPLIED in this pass

All correct-path scientific output is preserved; changes are crash-guards,
concurrency-safety, and redundant-work removal. Every edited Python file passes
`python3 -m py_compile`.

**HIGH**
- Parallel-well CSV race (`tracking_montage.py`) — both experiment-level CSVs
  (`_tracked_montage_summary.csv`, `_tracking-info.csv`) now append under an
  exclusive `fcntl` file lock via a new `_append_df_locked` helper; header
  written once (first writer). Filenames unchanged (consumed by
  `stable_cell_filter.py`, `overlay_montage.py`, `pipeline.nf`).
- Montage crash on incomplete tile sets (`montage.py`) — incomplete groups are
  now logged and skipped (`return None`) before the batch-update block, so
  `update_field`/`*_id`/`well`/`timepoint` can no longer be undefined.
- Per-label multiprocessing blowup (`tracking_montage.py`) — the montage mask is
  passed once per worker via a `Pool(initializer=...)` global instead of being
  re-pickled with every label.
- Alignment N-fold recompute + clobber + N+1 updates (`align_montage_dft.py`) —
  shifts computed once per `(well, timepoint)` and applied once per
  `(well, timepoint, channel)` montage; all sharing `tiledata` rows updated.
  (Verified in `montage.py` that all tiles of a well/tp/channel share one
  montage path, so this is redundancy removal, not a result change.)
- Swallowed DB delete failures (`segmentation_montage.py`) — now logged via
  `logger.error`.
- DB credentials (`sql.py`) — clear errors on missing/unreadable/malformed
  credential file; host/port/db/**user** now overridable via `GALAXY_DB_*`
  env vars (default unchanged). NOTE: the default is still the `postgres`
  superuser — an operator must still switch to a least-privilege role and move
  the secret off the shared plaintext CSV; that's a config/ops change, not code.

**MEDIUM**
- `sql.py add_row` — bulk inserts chunked (5000 rows) to avoid Postgres
  parameter-limit failures.
- `cv2.imread` None-checks added in `utils.py` (`find_max_dimensions`,
  `collapse_stack_to_image`), `align_montage_dft.py` (both loops), and
  `montage.py` (tile read wrapped in try/except → skip group).
- `align_montage_dft.py:132` — `.str.endswith('.tif', na=False)`.
- `utils.py` ImageMagick wrappers — check `returncode` + raise with stderr.
- `tracking_montage.py` — `cv2.imwrite` return value checked (raise on failure).
- `overlay_montage.py` — per-timepoint tasks now carry only the current well's
  summary rows instead of re-pickling the whole multi-well DataFrame (plus an
  empty-DataFrame guard).
- `db_util.py` — `get_raw_and_analysis_dir` checks `analysisdir` for None;
  `get_trackedmaskpath_from_other_channel` now scopes by experiment.
- `utils.py zerone_normalizer` — divide-by-zero guard for flat images.
- `run.sh` — driver `#SBATCH -N 2` → `-N 1` (driver only orchestrates).
- Added `finkbeiner.config.template` (referenced by `run.sh` +
  `setup-workspace.sh` but previously missing).

**LOW**
- `segmentation_montage.py` — scalar fallback threshold; `cv2.CV_AA` →
  `cv2.LINE_AA` in `utils.py`; `os.makedirs(..., exist_ok=True)` in `montage.py`;
  channel-prefix substitution now warns when no prefix matches
  (`tracking_montage.py`); `return_image_indices` no longer double-globs.

## DEFERRED (needs a design decision, benchmark, or a broader refactor)

Left untouched on purpose — flagged above with details:
- `sql.py` connection/reflection storm (reuse one `Database` per unit of work) —
  architectural; touches many call sites.
- `montage.py` peak-memory doubling (pre-allocate montage, write tiles in place)
  — restructures the hot loop; wants a memory benchmark.
- `tracking_montage.py` double mask read (`:691` vs `run`) — cache opportunity.
- `segmentation_montage.py` ThreadPool vs ProcessPool for CPU-bound work —
  needs a throughput benchmark before switching.
- `db_util.py get_trackedmaskpath_from_other_channel` — still not scoped by
  `welldata_id` (out of the method's current signature; currently unused).
- `segmentation_helper_montage.py` four near-duplicate update functions —
  consolidation deferred (risk of missing a caller).
- `utils.py` `pickle` cache → JSON, and `find_ind_of_filename` `Optional[int]`
  return type — behavior/return-type changes with caller risk.
