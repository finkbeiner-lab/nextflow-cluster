# Deploying the Pipeline on a VM

This guide covers running the Nextflow image-analysis pipeline (and, optionally,
the Visual Node Editor) on a single virtual machine — e.g. a cloud VM or an
on-prem VM outside the HPC cluster.

The pipeline was built for an HPC environment and makes three assumptions you
must satisfy (or replace) on a VM:

1. **Slurm** as the Nextflow executor (`nextflow.config`).
2. A **shared `/gladstone/finkbeiner` filesystem** for image data.
3. A reachable **PostgreSQL** database (`galaxy` on
   `fb-postgres01.gladstone.internal`) — every module reads/writes it.

There are two deployment shapes. Pick one:

- **Shape A — VM as a Slurm submit host.** The VM joins/points at an existing
  Slurm cluster. Nothing in the pipeline changes; this mirrors the cluster
  exactly. Best if you already have Slurm.
- **Shape B — Standalone VM (no Slurm).** The VM runs everything itself using
  Nextflow's `local` executor. Simplest to stand up; the VM must be sized for
  the whole workload since there is no fan-out to compute nodes.

---

## 1. Size the VM

Per-process defaults in `nextflow.config` are **15 GB RAM** and the bundled
per-well jobs ask for **4–17 CPUs**, with `maxForks = 22`. On a standalone VM
(Shape B) those forks all run locally.

| | Minimum (dev/small) | Recommended (real experiments) |
|---|---|---|
| vCPUs | 8 | 32–64 |
| RAM | 32 GB | 128–256 GB (montages of large plates are memory-hungry) |
| Disk | 200 GB SSD | 1 TB+ (raw tiles + montages + masks + Nextflow `work/`) |
| GPU | none (CPU segmentation) | 1× NVIDIA if using Cellpose (`--nv`) |
| OS | Ubuntu 22.04 / Rocky 8 | same |

Montage assembly currently holds all raw tiles **and** the assembled montage in
memory at once (see `CODE_REVIEW.md`), so err high on RAM for Shape B.

---

## 2. Install host prerequisites

```bash
# Apptainer (Singularity) — the pipeline runs inside a .sif container
# Ubuntu:
sudo add-apt-repository -y ppa:apptainer/ppa
sudo apt-get update && sudo apt-get install -y apptainer

# Java 17 + Nextflow (only needed on the host if you run nextflow outside the container)
sudo apt-get install -y openjdk-17-jre
curl -s https://get.nextflow.io | bash && sudo mv nextflow /usr/local/bin/

# GPU (only if using Cellpose): install the NVIDIA driver + container toolkit
```

> Security note: the container def and `install.sh` install Nextflow via
> `curl ... | bash` with no integrity check. The Visual Editor's def
> (`deepcell/deploy/visual-editor.def`) already pins a version + SHA-256 — do
> the same here for a hardened build.

---

## 3. Get the code and build the container

```bash
git clone <this-repo> nextflow-cluster
cd nextflow-cluster
apptainer build nextflow-cluster.sif nextflow-cluster.def   # needs root/fakeroot
```

Building the `.sif` needs `--fakeroot` or root. If the VM can't build it, build
on a machine that can and copy the `.sif` over (it is gitignored).

---

## 4. Provide the database

Every module imports `from sql import Database`, which reads a password from a
CSV and connects to `postgresql://postgres@fb-postgres01.gladstone.internal:5432/galaxy`
(`bin/sql.py`). You have two options:

**Option 1 — reach the existing DB.** Ensure the VM has network/DNS to
`fb-postgres01.gladstone.internal:5432` (VPN, peering, or `/etc/hosts` + firewall
rule) and that the credentials CSV exists at the path in
`Database.CREDENTIALS_PATH`. This keeps all data in the lab's central DB.

**Option 2 — run PostgreSQL on the VM.** Stand up a local Postgres, create the
`galaxy` database, and point the pipeline at it. This currently requires editing
`bin/sql.py` — the host, port, DB name, user, and credentials path are
**hardcoded** there. Recommended change (also called out in `CODE_REVIEW.md`):
read them from environment variables, e.g.

```python
host = os.environ.get("GALAXY_DB_HOST", "fb-postgres01.gladstone.internal")
user = os.environ.get("GALAXY_DB_USER", "galaxy_app")   # least-privilege, not postgres
db   = os.environ.get("GALAXY_DB_NAME", "galaxy")
```

Either way: **do not** keep using the `postgres` superuser with a plaintext,
world-readable password (see `CODE_REVIEW.md` §6).

```bash
# Local Postgres quickstart (Option 2)
sudo apt-get install -y postgresql
sudo -u postgres createuser --pwprompt galaxy_app
sudo -u postgres createdb -O galaxy_app galaxy
# The pipeline creates its own tables on first Database() init.
```

---

## 5. Provide the image data

The container bind-mounts `/gladstone/finkbeiner` read-write. On a VM you must
either:

- **Mount the real share** (NFS/SMB/VPN) at the same path `/gladstone/finkbeiner`
  so paths in the DB resolve unchanged, or
- **Stage data locally** and update the bind mounts + the `input_path` /
  `output_path` in your config to the local location.

The bind mounts live in `nextflow.config` (`containerOptions`). Update the two
`--bind` paths to match the VM:

```groovy
containerOptions = "--bind /path/on/vm/bin:/app \
                    --bind /path/on/vm/data:/path/on/vm/data:rw \
                    --nv"     // drop --nv if the VM has no GPU
```

Also update the hardcoded install path in `nextflow.config` (`container = ...`)
and in `run.sh` (`INSTALL_DIR=...`) to where you cloned the repo on the VM.

---

## 6. Choose the executor

### Shape A — VM is a Slurm submit host
Leave `nextflow.config` as-is (`executor = 'slurm'`, `queue = 'galaxy'`). Ensure
`sbatch` + munge work on the VM. Submit exactly like the cluster:

```bash
sbatch run.sh -c finkbeiner.config
```

### Shape B — Standalone VM (no Slurm)
Edit `nextflow.config` so Nextflow runs processes locally:

```groovy
process {
    executor = 'local'          // was 'slurm'
    // remove/ignore: queue, and drop 'time = 600h' (unused locally)
    cpus   = 4                  // per-process; tune to the VM
    memory = '15GB'
    maxForks = 4                // how many processes run at once ON THIS VM
    container = '/path/on/vm/nextflow-cluster/nextflow-cluster.sif'
    containerOptions = "--bind /path/on/vm/nextflow-cluster/bin:/app --bind /data:/data:rw --nv"
    shell = ['/bin/bash']
}
executor { cpus = 32; memory = '128 GB' }   // total pool the local executor may use
```

`maxForks = 22` is a cluster number — lower it for a VM or the box will thrash.
Then run without Slurm:

```bash
# run.sh assumes sbatch; on a standalone VM call Nextflow directly:
python3 bin/validate_config.py finkbeiner.config     # keep the config check
nextflow run pipeline.nf \
  -with-apptainer nextflow-cluster.sif \
  -c finkbeiner.config \
  -work-dir "$PWD/work_$$"
```

---

## 7. Configure and run

```bash
# Create your experiment config (copy the shipped one; there is no *.template yet)
cp finkbeiner.config my_run.config
# edit my_run.config: experiment name, wells, timepoints, channels, DO_* flags,
# and the input/output paths for the VM.

# Shape A:
sbatch run.sh -c my_run.config
# Shape B:
nextflow run pipeline.nf -with-apptainer nextflow-cluster.sif -c my_run.config -work-dir "$PWD/work_$$"
```

Outputs, logs, and Nextflow reports (`report.html`, `timeline.html`,
`trace.txt`) land in the launch directory. Progress: `tail -f nextflow.log`.

---

## 8. (Optional) Run the Visual Node Editor on the same VM

See `docs/VISUAL_EDITOR.md`. On a standalone VM without Slurm the editor's UI
boots (scheduler checks are warn-only) and you can design workflows, but it
cannot submit jobs until `sbatch`/munge are present — so the editor is best
paired with **Shape A**. For a no-Slurm VM, use the editor for design/DB
browsing and launch runs with the `nextflow run` command above.

---

## Deployment checklist

- [ ] VM sized for the workload (RAM especially, on Shape B)
- [ ] Apptainer installed; `nextflow-cluster.sif` built or copied in
- [ ] DB reachable (existing) **or** local Postgres created; `bin/sql.py` points at it
- [ ] Credentials moved off the plaintext superuser CSV (env var / least-priv role)
- [ ] Image data mounted at the expected path; `containerOptions` binds updated
- [ ] `INSTALL_DIR` / `container` paths in `run.sh` + `nextflow.config` updated
- [ ] Executor set (`slurm` for Shape A, `local` + tuned `maxForks` for Shape B)
- [ ] `python3 bin/validate_config.py <config>` passes
- [ ] A small test experiment completes end to end before scaling up
