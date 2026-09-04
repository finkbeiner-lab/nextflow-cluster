# Setting Up a Standalone Rocky 8 "Data Cruncher" for nextflow-cluster

A step-by-step runbook to turn a single Rocky 8 machine into a personal compute
node that runs the `nextflow-cluster` pipeline off the main Gladstone cluster,
with a real job queue so you can fire jobs from multiple projects at once.

Work top to bottom. Each stage ends with a **✅ Checkpoint** — don't move on
until it passes. Commands assume you have `sudo` on the box.

---

## 0. The plan (read this first)

You are rebuilding, on one machine, the four things the cluster gives the
pipeline:

| Dependency | On the cluster | On your box |
|---|---|---|
| **Scheduler** | Slurm (queue `galaxy`) | **Single-node Slurm** (same queue name) |
| **Container runtime** | Apptainer + `.sif` | Apptainer + copy of the `.sif` |
| **Data filesystem** | `/gladstone/finkbeiner` NAS | NFS-mounted from the same NAS |
| **Database** | Postgres `galaxy` @ `fb-postgres01` | Same DB, reached over the network |

**Why single-node Slurm instead of the `local` executor?** You said you'll send
jobs from several projects and want a queue. The `local` executor only queues
tasks *within one* `nextflow run`; two pipelines launched at once would both
grab the whole machine and oversubscribe it. Single-node Slurm gives you one
real queue that serializes and packs jobs from every project — and your existing
`nextflow.config` already targets `executor = 'slurm'`, so it works nearly as-is.

**GPU:** this box has an NVIDIA GPU, so we keep the container's `--nv` flag and
wire GPU into Slurm as a GRES resource (needed for Cellpose / CNN modules; the
core montage→segmentation→tracking→overlay workflow is CPU-only).

### Things to request from IT up front (do this first — they gate later stages)
These usually require IT and can take a day or two, so send the request now:

1. **NFS export** of `/gladstone/finkbeiner` to this host's IP (read/write), with
   matching UID/GID mapping so files aren't owned by `nobody`.
2. **Postgres access:** allow this host's IP to reach `fb-postgres01.gladstone.internal:5432`
   (firewall + a `pg_hba.conf` entry for the `galaxy` database).
3. A **static IP or DNS name** for the box (NFS/Postgres allow-lists key off it).
4. Confirm you have an account with `sudo`, and the **Galaxy DB password**
   (lab 1Password vault; also stored in `pass.csv` on the NAS — see Stage 7).

---

## 1. Connect and survey the machine

Open a terminal through ScreenConnect, then capture the box's specs — you'll
plug these numbers into the Slurm config later.

```bash
# Identity & OS
hostnamectl                       # confirm Rocky Linux 8.x
hostname -f                       # fully-qualified name (note this down)
ip -brief addr                    # note the primary IP

# Resources
nproc                             # CPU cores  -> Slurm CPUs
free -h                           # RAM        -> Slurm RealMemory
lsblk -f                          # disks; find a big partition for scratch
df -hT /                          # free space on root

# GPU (should list your NVIDIA card)
lspci | grep -i nvidia
nvidia-smi 2>/dev/null || echo "NVIDIA driver not yet installed (Stage 5)"
```

**✅ Checkpoint:** You have the FQDN, IP, core count, and RAM written down, and
you can see the NVIDIA card in `lspci`.

---

## 2. Base system packages

```bash
sudo dnf -y update
sudo dnf -y install epel-release
sudo dnf -y install \
    git curl wget nano vim tar which \
    gcc gcc-c++ make \
    nfs-utils \
    bind-utils nmap-ncat \
    python3 python3-pip
```

Set a sane timezone/hostname if needed:

```bash
sudo timedatectl set-timezone America/Los_Angeles
# sudo hostnamectl set-hostname cruncher01.gladstone.internal   # only if IT asks
```

**✅ Checkpoint:** `git --version`, `python3 --version`, and `nc -h` all work.

---

## 3. Java (required by Nextflow)

Your pipeline pins Nextflow **24.04.4**, which runs on Java 17.

```bash
sudo dnf -y install java-17-openjdk java-17-openjdk-devel
java -version                      # should report 17.x
```

Make it the default if multiple JDKs are present:

```bash
sudo alternatives --config java    # pick the 17 entry
```

**✅ Checkpoint:** `java -version` prints `17`.

---

## 4. Nextflow

Install to `/opt/nextflow` so it's on a shared, predictable path (matches the
launcher search in `run.sh`).

```bash
sudo mkdir -p /opt/nextflow
cd /opt/nextflow
sudo bash -c 'curl -s https://get.nextflow.io | bash'
sudo chmod 755 /opt/nextflow/nextflow
sudo ln -sf /opt/nextflow/nextflow /usr/local/bin/nextflow
```

Pin the tested engine version system-wide so every job uses the same one
(the cluster hit real breakage when unpinned jobs grabbed Nextflow ≥25):

```bash
echo 'export NXF_VER=24.04.4' | sudo tee /etc/profile.d/nextflow.sh
sudo chmod +x /etc/profile.d/nextflow.sh
source /etc/profile.d/nextflow.sh
nextflow -version                  # should show 24.04.4
```

**✅ Checkpoint:** `nextflow -version` reports **24.04.4**.

---

## 5. Apptainer (Singularity) + GPU

### 5a. Apptainer

```bash
sudo dnf -y install apptainer
apptainer --version
```

### 5b. NVIDIA driver (skip if `nvidia-smi` already worked in Stage 1)

```bash
sudo dnf config-manager --add-repo \
  https://developer.download.nvidia.com/compute/cuda/repos/rhel8/x86_64/cuda-rhel8.repo
sudo dnf -y module install nvidia-driver:latest-dkms
sudo reboot            # reconnect via ScreenConnect after it comes back
```

After reboot:

```bash
nvidia-smi             # must list your GPU and a driver version
```

### 5c. Prove Apptainer can see the GPU

```bash
apptainer exec --nv docker://nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

**✅ Checkpoint:** `nvidia-smi` **inside** the `--nv` container lists your GPU.
(If the driver install is fussy, you can proceed with CPU-only modules and fix
the GPU later — just don't run Cellpose/CNN until this passes.)

---

## 6. Mount the NAS (`/gladstone/finkbeiner`)

The pipeline reads and writes everything under `/gladstone/finkbeiner`, and the
**database password lives there** (`/gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv`),
so this mount is non-negotiable.

**Find the NFS source.** On any working cluster node run:

```bash
mount | grep gladstone         # shows e.g.  nas-server:/finkbeiner on /gladstone/finkbeiner
```

Note the `server:/export` on the left — that's your source. Then on the box:

```bash
sudo mkdir -p /gladstone/finkbeiner
# Quick test mount (replace with the real source from above):
sudo mount -t nfs <nas-server>:/<export-path> /gladstone/finkbeiner
ls /gladstone/finkbeiner        # should list lab directories
```

Make it permanent via `/etc/fstab`:

```bash
echo '<nas-server>:/<export-path>  /gladstone/finkbeiner  nfs  defaults,_netdev,rw  0 0' \
  | sudo tee -a /etc/fstab
sudo mount -a                    # remount everything; no errors = good
```

> If files show as owned by `nobody:nobody` or writes fail, that's a UID/GID
> mapping or export-permission issue — go back to IT with your host IP and ask
> them to align the export's `root_squash`/idmap with the cluster.

**✅ Checkpoint:**
```bash
cat /gladstone/finkbeiner/lab/GALAXY_INFO/pass.csv   # you can READ the creds file
touch /gladstone/finkbeiner/<your-lab-dir>/.write_test && echo "write OK"
```

---

## 7. Database connectivity

The code (`bin/sql.py`) reads the password from `pass.csv` on the NAS and
defaults the host to `fb-postgres01.gladstone.internal:5432`. It also honors
env overrides (`GALAXY_DB_HOST`, `GALAXY_DB_USER`, `GALAXY_DB_PORT`) — handy if
IT gives you a dedicated role instead of the `postgres` superuser.

Test that the box can reach the DB port:

```bash
nc -vz fb-postgres01.gladstone.internal 5432
```

- **`succeeded`** → connectivity is good, nothing more to do here.
- **`timed out` / `refused`** → IT still needs to open the firewall and add a
  `pg_hba.conf` line for this host's IP to the `galaxy` DB. Send them the IP.

Optional end-to-end check from inside the container (uses your real code):

```bash
apptainer exec \
  --bind /gladstone/finkbeiner:/gladstone/finkbeiner:rw \
  --bind /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/bin:/app \
  /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/nextflow-cluster.sif \
  python3 -c "import sys; sys.path.insert(0,'/app'); from sql import Database; Database(); print('DB OK')"
```

If you were issued a non-default role, export the overrides before running
(and add them to your config env later):

```bash
export GALAXY_DB_USER=<your_role>
export GALAXY_DB_HOST=fb-postgres01.gladstone.internal
export GALAXY_DB_PORT=5432
```

**✅ Checkpoint:** `nc -vz ... 5432` succeeds, and (ideally) the `DB OK` probe
prints.

---

## 8. Single-node Slurm (your job queue)

This gives you one queue named `galaxy` — matching your config — that serializes
and packs jobs from every project.

### 8a. Install

```bash
sudo dnf -y install munge munge-libs
sudo dnf -y install slurm slurm-slurmctld slurm-slurmd slurm-perlapi
```

### 8b. Munge auth key

```bash
sudo /usr/sbin/create-munge-key -r    # or: sudo dd if=/dev/urandom of=/etc/munge/munge.key bs=1 count=1024
sudo chown munge:munge /etc/munge/munge.key
sudo chmod 400 /etc/munge/munge.key
sudo systemctl enable --now munge
munge -n | unmunge | grep STATUS      # STATUS: Success (0)
```

> If munge fails with a logfile permission error, it must run as the `munge`
> user, not root: `sudo chown -R munge:munge /var/log/munge /var/lib/munge /etc/munge`.

### 8c. Generate the node line and write `slurm.conf`

Let Slurm detect the exact CPU/memory line for this box:

```bash
slurmd -C            # prints:  NodeName=... CPUs=.. Boards=.. ... RealMemory=..
```

Create `/etc/slurm/slurm.conf` (replace `HOSTNAME` with `hostname -s`, and paste
the `CPUs=`/`RealMemory=` numbers from `slurmd -C`; set `RealMemory` a bit below
physical RAM so the OS keeps headroom):

```ini
ClusterName=cruncher
SlurmctldHost=HOSTNAME

AuthType=auth/munge
SlurmUser=slurm
StateSaveLocation=/var/spool/slurmctld
SlurmdSpoolDir=/var/spool/slurmd
SlurmctldPidFile=/var/run/slurmctld.pid
SlurmdPidFile=/var/run/slurmd.pid
SlurmctldLogFile=/var/log/slurmctld.log
SlurmdLogFile=/var/log/slurmd.log

ProctrackType=proctrack/linuxproc
TaskPlugin=task/none
SchedulerType=sched/backfill
SelectType=select/cons_tres
SelectTypeParameters=CR_Core_Memory
ReturnToService=2

# --- GPU as a schedulable resource (needed for Cellpose/CNN) ---
GresTypes=gpu

# --- This machine (paste CPUs/RealMemory from `slurmd -C`) ---
NodeName=HOSTNAME CPUs=XX RealMemory=XXXXX Gres=gpu:1 State=UNKNOWN

# --- One queue named to match nextflow.config ---
PartitionName=galaxy Nodes=ALL Default=YES MaxTime=INFINITE State=UP
```

Tell Slurm about the GPU device in `/etc/slurm/gres.conf`:

```ini
NodeName=HOSTNAME Name=gpu File=/dev/nvidia0
```

### 8d. Permissions, service start

```bash
sudo mkdir -p /var/spool/slurmctld /var/spool/slurmd
sudo chown slurm: /var/spool/slurmctld
sudo systemctl enable --now slurmctld slurmd

sinfo                                 # partition 'galaxy', node in 'idle'
srun hostname                         # runs a command through the queue
srun --gres=gpu:1 nvidia-smi -L       # confirms GPU scheduling
```

If the node shows `drain`/`down`, bring it up:

```bash
sudo scontrol update NodeName=HOSTNAME State=RESUME
```

**✅ Checkpoint:** `sinfo` shows the `galaxy` partition with your node `idle`,
and `srun hostname` returns the hostname.

---

## 9. Get the pipeline code and container onto the box

You have two clean options.

**Option A — run straight from the NAS (simplest, always in sync).**
The install dir already exists on the mounted NAS, so you can point at it
directly and skip copying anything:

```bash
ls /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/nextflow-cluster.sif
```

**Option B — keep a local copy** (isolates you from cluster changes; put it on a
fast local disk):

```bash
mkdir -p ~/nextflow-cluster
rsync -av --progress \
  /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/ \
  ~/nextflow-cluster/
```

Either way, confirm the container is intact:

```bash
apptainer inspect <path>/nextflow-cluster.sif    # prints labels, no error
```

> Rebuilding the `.sif` from `nextflow-cluster.def` is possible but needs root
> and ~10+ minutes; only do it if the prebuilt one is missing or stale.

**✅ Checkpoint:** You can `ls` the `.sif` and `apptainer inspect` it cleanly.
Decide now whether "install dir" = the NAS path (A) or `~/nextflow-cluster` (B).

---

## 10. Adapt the config to this box

Two files reference cluster-specific paths and sizes. Copy them so you keep the
originals, then edit.

### 10a. `nextflow.config` — resources and scratch

Your current `process` block requests `executor = 'slurm'`, `queue = 'galaxy'`
— **keep both**, they match the Slurm you just built. Adjust only what's
machine-specific:

- **`workDir`** (line ~25): currently a cluster path. Point it at fast local
  storage (falls back to NAS if you have no big local disk):
  ```groovy
  workDir = '/data/nf-work'          // create it: sudo mkdir -p /data/nf-work && sudo chown $USER /data/nf-work
  ```
- **`maxForks`** (line ~88): cap concurrency to your core count. On one box,
  something like `maxForks = <cores / cpus-per-task>` (e.g. 8 on a 32-core box
  with 4-CPU tasks). Start conservative.
- **`memory` / `cpus`**: leave the per-task defaults, but make sure
  `maxForks × memory` stays under the box's RAM, or Slurm will hold jobs
  (which is exactly the queueing you want, but keep it intentional).
- **`executor.queueSize`** (line ~71): fine as-is; it just caps how many tasks
  Nextflow keeps in the Slurm queue.
- The `--nv` bind flag stays (you have a GPU). Keep the
  `--bind /gladstone/finkbeiner:/gladstone/finkbeiner:rw`.

### 10b. `run.sh` — install dir and (optional) DB overrides

- Set **`INSTALL_DIR`** (line 18) to your choice from Stage 9:
  - Option A: leave the existing NAS path.
  - Option B: `INSTALL_DIR="$HOME/nextflow-cluster"`.
- The launcher already searches `/usr/local/bin/nextflow` (Stage 4) and pins
  `NXF_VER=24.04.4`, so those just work.
- If IT gave you a dedicated DB role, add near the top of `run.sh`:
  ```bash
  export GALAXY_DB_USER=<your_role>
  export GALAXY_DB_HOST=fb-postgres01.gladstone.internal
  export GALAXY_DB_PORT=5432
  ```

### 10c. Per-project config

For each project, copy the template and edit experiment/wells/modules:

```bash
cp <INSTALL_DIR>/finkbeiner.config.template ~/projects/<proj>/finkbeiner.config
# edit input_path, output_path, experiment, DO_* flags, wells/timepoints/channels
```

**✅ Checkpoint:** `nextflow.config` has a writable `workDir`, a sane
`maxForks`, and `run.sh`'s `INSTALL_DIR` points where your code actually lives.

---

## 11. First test run

Pick a **small** experiment (a handful of wells, `DO_STD_WORKFLOW` or the IXM
variant) so a failure surfaces fast.

```bash
cd ~/projects/<proj>
sbatch <INSTALL_DIR>/run.sh -c ./finkbeiner.config

squeue                      # watch it queue then run
tail -f *_<exp>.out         # the run.sh log (name includes user/date/exp)
```

When it finishes, review the reports Nextflow drops in the launch dir:
`report.html`, `timeline.html`, `trace.txt`, `dag.html`.

**✅ Checkpoint:** The run reaches "Nextflow pipeline completed", and spot-check
that rows appear in the DB / outputs land under your `output_path`.

---

## 12. Day-to-day: sending jobs from multiple projects

This is the payoff. Each project is just its own directory + config; `sbatch`
each one and Slurm queues them into the single `galaxy` partition:

```bash
cd ~/projects/projA && sbatch <INSTALL_DIR>/run.sh -c ./finkbeiner.config
cd ~/projects/projB && sbatch <INSTALL_DIR>/run.sh -c ./finkbeiner.config
cd ~/projects/projC && sbatch <INSTALL_DIR>/run.sh -c ./finkbeiner.config
squeue                       # all three tracked; Slurm packs them to fit the box
```

Useful controls:

- `squeue` — what's running/pending. `scancel <jobid>` — kill one.
- `sinfo` — node/partition state. `scontrol show job <jobid>` — details.
- Concurrency is bounded by **Slurm** (cores/RAM available on the node) and by
  each run's **`maxForks`**. Jobs beyond capacity wait in `PENDING` — that's the
  queue doing its job, not an error.
- To reserve headroom (e.g. keep the GPU free for one project), request
  `--gres=gpu:1` only in the runs that need it; CPU-only runs won't block on it.

---

## Troubleshooting

| Symptom | Cause & fix |
|---|---|
| `nextflow: command not found` (exit 127) | Launcher not on PATH for the batch shell. You symlinked to `/usr/local/bin/nextflow` in Stage 4; verify `ls -l /usr/local/bin/nextflow`. Or run `sbatch --export=NEXTFLOW_BIN=/opt/nextflow/nextflow ...`. |
| Config parse error re: `def DEEPCELL_DEV` | A Nextflow ≥25 engine slipped in. Confirm `echo $NXF_VER` = `24.04.4`; it's pinned in `/etc/profile.d/nextflow.sh`. |
| `mkdir ... NXF_HOME ... Permission denied` | Point it somewhere writable: `export NXF_HOME=$HOME/.nxf` (or set in `run.sh`). |
| `Database credentials file not found: .../pass.csv` | NAS not mounted or export missing that dir. Re-check Stage 6 (`mount -a`, `ls .../GALAXY_INFO/pass.csv`). |
| DB connect hangs/refused | Firewall / `pg_hba.conf` not opened for this host. Re-run `nc -vz fb-postgres01... 5432`; send IT your IP (Stage 7). |
| Node shows `down`/`drain` in `sinfo` | `sudo scontrol update NodeName=<host> State=RESUME`; check `/var/log/slurmd.log`. |
| munge `STATUS: ...` not Success | Key perms/ownership: `sudo chown -R munge:munge /etc/munge /var/lib/munge /var/log/munge`, then `systemctl restart munge`. |
| GPU tasks fail / not visible | `nvidia-smi` on host, then `apptainer exec --nv ... nvidia-smi`; confirm `gres.conf` device path matches `/dev/nvidia0`. |
| Files owned by `nobody`, writes fail on NAS | NFS idmap/squash mismatch — IT must align the export with cluster UID/GID for your host. |

---

## Quick reference — the four dependencies

```
Scheduler : single-node Slurm, partition "galaxy"   ->  sinfo / squeue / sbatch
Container : apptainer + nextflow-cluster.sif (--nv) ->  apptainer inspect <sif>
Filesystem: NFS mount /gladstone/finkbeiner          ->  mount -a ; ls .../GALAXY_INFO
Database  : fb-postgres01.gladstone.internal:5432    ->  nc -vz fb-postgres01... 5432
```

Launch pattern (from any project dir):
```bash
sbatch <INSTALL_DIR>/run.sh -c ./finkbeiner.config
```
