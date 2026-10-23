# DEVELOPMENT.md — where the code lives and how a change reaches a job

This document maps the development topology for this pipeline: the several places
the code exists at once, how an edit propagates to a running Slurm job, and the
gotchas that only appear on the real cluster path. It exists because "I edited the
file but the job ran the old code" has happened more than once — almost always
because one of the copies below was out of sync.

## The four copies of the code

The same Python/Nextflow code exists in up to four places simultaneously. Keeping
them straight is the whole game.

| # | Location | What it is | How it updates |
|---|----------|------------|----------------|
| 1 | **Mac working copy** — `~/Desktop/nextflow-cluster` | Where you edit. | You edit it. |
| 2 | **GitHub** — `finkbeiner-lab/nextflow-cluster` | The shared source of truth. | `git push` from the Mac. |
| 3 | **Cluster checkout(s)** — `~/nextflow-cluster` (+ worktrees) on the NAS | What `run.sh` reads (`pipeline.nf`, `modules.nf`, `bin/`), and what the container bind-mounts in dev mode. | `git pull` on the cluster. |
| 4 | **Container `/app` + `/ml`** — baked inside `nextflow-cluster.sif` | What the Python actually runs against inside Singularity. | **Rebuild the SIF** (`apptainer build`), OR shadow it with a dev bind (below). |

A change on the Mac is invisible to a cluster job until it has travelled **1 → 2
→ 3**, and then into **4** either by a rebuild or a dev bind. Skipping any step is
the usual cause of "it ran the old code."

## The one-liner mental model

```
edit on Mac  →  git push  →  git pull on cluster  →  (dev bind OR rebuilt SIF)  →  sbatch run.sh
```

## Reaching the cluster

All cluster commands go over ScaleFT:

```bash
sft ssh --command '<command>' fb-galaxy-dev01
```

To run a script in-container ad hoc (base64 avoids quoting hell):

```bash
base64 <<'EOF' | sft ssh --command 'base64 --decode | bash' fb-galaxy-dev01
SIF=~/nextflow-cluster/nextflow-cluster.sif
apptainer exec --nv -B /gladstone/finkbeiner:/gladstone/finkbeiner "$SIF" python - <<'PY'
...python...
PY
EOF
```

The dev node `fb-galaxy-dev01` has **no GPU** — GPU work must go through `sbatch`
to the `galaxy` partition (V100 nodes `fb-gpu-compute01/02`). Do not run heavy
compute on the dev node; container builds are the exception (that is where
`apptainer build` runs).

## Branches and worktrees

The cluster checkout is a normal git repo, so it is on exactly **one branch at a
time**. This project juggles several personal branches (e.g. `austin/RGEDI`,
`austin/neurite-module`). Switching the single `~/nextflow-cluster` checkout
between them means a job can run against a branch that lacks the file you expect
— e.g. a montage job died with "can't open `/app/neurite_montage.py`" because the
checkout was on `austin/RGEDI`, which predates that module.

**Fix: one git worktree per branch**, so branches coexist without switching:

```bash
cd ~/nextflow-cluster
git worktree add ~/nextflow-cluster-neurite austin/neurite-module
```

Now `~/nextflow-cluster` can stay on `austin/RGEDI` while
`~/nextflow-cluster-neurite` holds `austin/neurite-module`. Point runs at the
worktree with `NEXTFLOW_INSTALL_DIR` (below). **Before any cluster run, confirm
which branch/dir the code is in** (`git -C <dir> branch --show-current`).

The SIF is a large built artifact, not tracked by git, so it does not appear in a
new worktree. Either symlink the existing one in or build a fresh one there:

```bash
ln -sfn ~/nextflow-cluster/nextflow-cluster.sif ~/nextflow-cluster-neurite/nextflow-cluster.sif
```

## Env-var overrides (no tracked-file edits needed)

`run.sh` and `nextflow.config` default to the shared canonical install but honor
env vars so a branch/worktree run needs no edits to tracked files:

| Env var | Effect |
|---------|--------|
| `NEXTFLOW_INSTALL_DIR=<dir>` | `run.sh` reads `pipeline.nf` / `modules.nf` / the SIF from `<dir>`; the DEV bind uses `<dir>/bin`. Default = shared steve install. |
| `DEEPCELL_DEV=1` | Bind-mount the host `bin/` over the container's baked `/app`, so a `bin/*.py` edit is used without rebuilding the SIF. Default = off (use baked `/app`). |
| `DEEPCELL_DEV_BIN=<dir>` | Explicit override of which `bin/` the DEV bind uses (rarely needed; defaults to `NEXTFLOW_INSTALL_DIR/bin`). |
| `NEXTFLOW_BIN`, `NXF_VER`, `NXF_HOME` | Nextflow launcher / version / cache overrides (see `run.sh` header comments). |

## Dev mode vs. baked (the important distinction)

There are two ways the container gets the current Python:

- **Dev mode (`DEEPCELL_DEV=1`):** the host `bin/` is bind-mounted over `/app`,
  shadowing whatever is baked. Fast iteration — `git pull` on the cluster, and the
  next task uses the new code with no rebuild. Use this while developing.
- **Prod mode (default):** the container runs its **baked** `/app` (and `/ml`).
  Reproducible and self-contained. A `bin/`/`ml/` change is invisible until you
  **rebuild the SIF**.

**Gotcha — per-process `containerOptions` replaces the global one.** A Nextflow
process that sets its own `containerOptions` (the GPU processes: `CELLPOSE`,
`NEURITE_MONTAGE`) does **not** inherit `nextflow.config`'s global options,
including the `DEEPCELL_DEV` `/app` bind and `--nv`. So each such process must
repeat `--nv` itself, and to work in dev mode it must add the `/app` bind itself.
`NEURITE_MONTAGE` does this via a closure; `CELLPOSE` does not, so `CELLPOSE`
changes require a rebuild.

## Rebuilding the container (bake `bin/` + `ml/` into the SIF)

Run from the checkout/worktree whose code you want baked. Build to a fresh file,
then swap the symlink (keeps the old SIF as instant rollback). Detached, since it
takes ~20–40 min and needs the network:

```bash
sft ssh --command 'cd ~/nextflow-cluster-neurite && nohup bash -c "apptainer build --fakeroot nextflow-cluster.new.sif nextflow-cluster.def && ln -sfn \$PWD/nextflow-cluster.new.sif nextflow-cluster.sif && echo BAKE_OK" > ~/bake.log 2>&1 &' fb-galaxy-dev01
sft ssh --command 'tail -15 ~/bake.log' fb-galaxy-dev01   # look for BAKE_OK
```

`nextflow-cluster.def` `%files` bakes `bin → /app`, `ml → /ml`, plus
`modules.nf` / `pipeline.nf`; `%environment` sets `NEURITE_ML_DIR=/ml/neurite`.

## Running the pipeline

Dev mode (no rebuild; uses the worktree bin via the bind):

```bash
cd ~/nextflow-cluster-neurite
sbatch --export=ALL,DEEPCELL_DEV=1,NEXTFLOW_INSTALL_DIR=$HOME/nextflow-cluster-neurite \
       run.sh -c <your.config>
```

Prod mode (module baked into the SIF; no `DEEPCELL_DEV`):

```bash
cd ~/nextflow-cluster-neurite
sbatch --export=ALL,NEXTFLOW_INSTALL_DIR=$HOME/nextflow-cluster-neurite \
       run.sh -c <your.config>
```

`run.sh` submits a lightweight **CPU launcher** that runs Nextflow; Nextflow then
submits each process (e.g. `NEURITE_MONTAGE`) as its **own** Slurm job, which
reserves its GPU via `clusterOptions`. So a GPU run shows two jobs: the launcher
(`nextflow-run`, CPU) and the child (`nf-NEURITE_MONTAGE_*`, `gres/gpu:1`). The
launcher writes `<launchdir>/<user>_<date>_<jobid>_<exp>.out`.

## Gotchas checklist (things that only bite on the real path)

- **Executable bit on new `bin/` scripts.** Nextflow invokes them bare
  (`neurite_montage.py …`) via the shebang + PATH, so a script committed `644`
  fails with exit **126 "Permission denied"** (a standalone `python script.py`
  test never catches this). `chmod +x` and commit the mode.
- **Cellpose bf16 on the V100.** Cellpose 4.x defaults to `use_bfloat16=True`,
  which the V100 (no native bf16) emulates ~10× slower. Always build the model
  with `use_bfloat16=False`. (See the `cellpose-bf16-v100` note.)
- **Plate-map CSV validation is unconditional.** `pipeline.nf` validates
  `chosen_wells` against `params.platemap_path` (a CSV with a `well` column), not
  the DB — point it at a platemap that actually contains your wells.
- **`nextflow.config` `workDir` is hardcoded** to a shared path; `run.sh`
  overrides it with `-work-dir`, so real runs are fine, but a manual
  `nextflow run` needs `-w <writable-dir>`.
- **Which GPU?** Log `torch.cuda.get_device_name(0)`. `galaxy` = V100; the `Kif`
  cluster = H100 (bf16-native). Non-GPU jobs sometimes squat all cores on the GPU
  nodes and block GPU jobs — check `sinfo -N -p galaxy` and who is running there.
