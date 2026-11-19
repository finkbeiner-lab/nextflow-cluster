---
name: cluster-batch-submit
description: >-
  The division of labor for sending jobs to the Finkbeiner Slurm cluster: Claude
  writes the self-contained sbatch script, the user submits it. Use this skill
  WHENEVER a cluster job is ready to run, or the user says something is "ready to
  submit", "let's run this on the cluster / on galaxy / on Slurm", or asks how to
  submit, queue, or watch a batch job. Do NOT run sbatch or scancel yourself:
  hand the user the submit, queue, and log commands.
---

# Cluster Batch Submit

## Who does what

Claude **writes** the `.sbatch` script and may **read** data, logs and results over
`sft` (read-only). The user **submits**, **cancels**, and decides what to delete.
Never run `sbatch` or `scancel` yourself. The dev node `fb-galaxy-dev01` is shared and
has no GPU: never run heavy work on it. All compute goes through Slurm on the
`galaxy` partition.

## Writing the script

- Make it self-contained. Either call code from the user's checkout on the cluster
  (for example `~/nextflow-cluster`, check the branch first), or embed Python modules
  and small manifests in the script as base64 heredocs that a `mktemp -d` folder
  receives at job start (`trap 'rm -rf "$WORK"' EXIT`).
- Run Python inside the container:
  `apptainer exec [--nv] -B /gladstone/finkbeiner:/gladstone/finkbeiner <SIF> python ...`.
  Use `--nv` and `#SBATCH --gres=gpu:1` only for GPU work.
- Use a relative `#SBATCH --output=<name>_%j.out` so the log lands in the folder the
  user submits from. Default `#SBATCH --time=05:00:00` unless the job needs more.
- Outputs go ONLY to a lab project folder on the HIVE (`/gladstone/finkbeiner/<lab>/...`),
  never to the cluster home. Raw data under `/gladstone/finkbeiner/robodata/` is read-only.
- Save the script on the user's computer in `~/sbatch-claude/` (ask if they use another folder).

## What to hand the user when a script is ready

Fill in `<project>` (a folder in the cluster home for scripts and logs; ask the user
which one) and `<file>`. Give each command in its own `bash` code block.

1. Submit:
```bash
base64 < ~/sbatch-claude/<file>.sbatch | sft ssh --command 'mkdir -p ~/<project> && cd ~/<project> && base64 --decode > <file>.sbatch && sbatch <file>.sbatch' fb-galaxy-dev01
```
2. Queue (whole queue, so the user can see GPU contention):
```bash
sft ssh --command 'squeue -o "%.8i %.10u %.14j %.9P %.8T %.10M %.9l %.6D %.10b %R" -S -t,i' fb-galaxy-dev01
```
3. Log:
```bash
sft ssh --command 'cat ~/<project>/<name>_*.out' fb-galaxy-dev01
```

`base64` turns the script into plain text so it passes through the SSH pipe without
quoting or newline problems. If `sft` says "Authentication token expired", the user
must run `sft unenroll` and then `sft enroll` in a browser; do not try to work around it.

## Notes

- The `galaxy` partition has only 2 GPU nodes (one V100 each). A GPU job that is
  `PENDING (Resources)` is usually waiting behind other users' jobs, not broken.
- Cellpose 4 on the V100s: always `use_bfloat16=False` (about 10x faster).
