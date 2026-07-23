# Spinning Up the Visual Node Editor ("Deep Cell")

The Visual Node Editor is the web app (React frontend + FastAPI backend) that
lets you build, save, run, and monitor the Nextflow image-analysis pipeline
from a browser instead of hand-editing `finkbeiner.config` and running
`sbatch`. Its source lives in `deepcell/` in this checkout and is deployed to
`.../nextflow-cluster/deepcell/` on the cluster.

This page has three sections:
1. **Local dev** — run it on your laptop/workstation to try it or develop.
2. **Cluster / server** — run it as a supervised Apptainer instance (production).
3. **Where things live & how to check health.**

For the full operator runbook (upgrades, rollback, DB restore, TLS, auth) see
`deepcell/deploy/DEPLOYMENT.md`. This page is the quick "how do I turn it on".

---

## 1. Local development (laptop / workstation)

Good for trying the UI or editing code. No Slurm and no PostgreSQL are required
— the backend boots in fixture mode (`EDITOR_DISABLE_PG=1`) and the scheduler
is warn-only, so you can design workflows but not actually submit cluster jobs.

**Prerequisites:** Python 3.11+, Node.js 20+.

### Backend

```bash
cd deepcell/backend
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .            # if pyproject.toml is present
# There is a helper for dev:
bash run-dev.sh             # starts uvicorn with reload on 127.0.0.1:8000
```

If you prefer to run it by hand:

```bash
export PYTHONPATH=src
export EDITOR_DISABLE_PG=1                       # fixture experiments, no DB
export EDITOR_CATALOG_DIR="$PWD/catalog"
export EDITOR_TEMPLATES_DIR="$PWD/templates"
uvicorn finkbeiner_editor.main:app --reload --port 8000
```

### Frontend

```bash
cd deepcell/frontend
npm ci
npm run dev                 # Vite dev server, usually http://localhost:5173
```

The Vite dev server proxies `/api` to the backend on `:8000`. Open the URL Vite
prints. Health check: `curl http://127.0.0.1:8000/health` → `{"status":"ok"}`.

---

## 2. Cluster / server (production, Apptainer + systemd)

This is how it runs for real: a single Apptainer (Singularity) image, nginx +
gunicorn inside it, supervised by systemd, submitting jobs to Slurm via the
host's `sbatch`.

### Prerequisites on the host

| Requirement | Check |
|---|---|
| `apptainer` ≥ 1.2 | `apptainer --version` |
| `munge` + `munged` running | `systemctl is-active munge` → `active` |
| Slurm CLIs at `/usr/bin/{sbatch,squeue,scancel,sacct,sinfo}` | `test -x /usr/bin/sbatch` |
| Node.js 20+ (build time only) | `npm --version` |
| TLS cert + key | `/etc/ssl/deepcell.crt` and `.key` |
| Service account `finkbeiner-svc` | see `deploy/DEPLOYMENT.md` §1.5 |

### Build the image

```bash
cd deepcell            # (== .../nextflow-cluster/deepcell on the cluster)
( cd frontend && npm ci && npm run build )   # produces frontend/dist
bash deploy/build.sh                          # -> deepcell.sif + .sha256
```

`build.sh` builds the React bundle (if `npm` is on PATH), then runs
`apptainer build` and records a SHA-256 next to the image so you can verify and
roll back later.

### Start it

**One-shot (manual):**

```bash
bash deploy/apptainer-instance.sh            # binds Slurm CLIs, munge, data tree
apptainer instance list                       # should list "deepcell"
```

**Supervised (recommended):**

```bash
sudo cp deploy/deepcell.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now deepcell
sudo systemctl status deepcell
```

The launcher (`apptainer-instance.sh`) is hardened: `--cleanenv`, `--no-home`,
`--writable-tmpfs`, `--no-init`, the `/gladstone/finkbeiner` tree, the Slurm
CLIs and munge socket bound in, and a persisted session secret. It is
idempotent — it stops any prior instance of the same name first.

### Reach it

The in-container nginx serves the app on **`https://<host>:8080/`**. To submit
real jobs the container needs `sbatch` + munge (bound by the launcher);
`EDITOR_DISABLE_PG=0` switches the Database tab from fixtures to the live
`galaxy` PostgreSQL.

### Upgrade / roll back

```bash
# upgrade: rebuild and restart
( cd frontend && npm ci && npm run build ) && bash deploy/build.sh
sudo systemctl restart deepcell

# roll back: promote an archived .sif (see deploy/DEPLOYMENT.md §6)
```

---

## 3. Health, logs, and where things live

```bash
# Is it up?
apptainer instance list
ss -ltnp | grep 8080
curl -k https://localhost:8080/health          # {"status":"ok"}

# Logs (systemd runs gunicorn in the foreground -> journal)
journalctl -u deepcell -f
apptainer instance logs deepcell          # stdout
apptainer instance logs --err deepcell    # stderr
```

| Thing | Location |
|---|---|
| Container image | `.../deepcell/deepcell.sif` (+ `.sha256`) |
| Editor state (SQLite, workflows, runs) | `EDITOR_DATA` (default `.../deepcell/data`, `0700`) |
| Nextflow home | `NXF_HOME` (default `.../deepcell/nxf`) |
| Session secret | `<EDITOR_DATA>/.session_secret` (`0600`) |
| Deploy scripts / def / nginx | `deepcell/deploy/` |

## Common problems

| Symptom | Cause / fix |
|---|---|
| Unit fails instantly | munge socket missing → `sudo systemctl restart munge` (`deploy/DEPLOYMENT.md` §4.1) |
| `WARNING: /usr/bin/sbatch not found` | Slurm client not installed / not at `/usr/bin`; jobs can't submit (§4.2) |
| `bind() to 0.0.0.0:8080 failed` | Port in use or stale instance → `apptainer instance stop deepcell` (§4.3) |
| Build fails | `/var/tmp` full (`APPTAINER_TMPDIR=`), or `frontend/dist` missing (`npm run build`) (§4.4) |
| Database tab shows fixtures only | `EDITOR_DISABLE_PG=1`; export `EDITOR_DISABLE_PG=0` to use live PostgreSQL |
