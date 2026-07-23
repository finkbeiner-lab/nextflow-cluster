#!/bin/bash

# Slurm job options
#SBATCH --job-name=nextflow-run
#SBATCH --time=7-00:00:00
#SBATCH -N 1
#SBATCH --output=/dev/null
##SBATCH --gres=gpu:v100:1
#SBATCH --distribution=block:block

# ---------------------------------------------------------------------------
# INSTALL_DIR: where the shared pipeline code, scripts, and container live.
# USER_DIR:    where the user's finkbeiner.config lives and logs are written.
#
# Users run:  sbatch /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/run.sh
# from their own workspace directory.
# ---------------------------------------------------------------------------
INSTALL_DIR="/gladstone/finkbeiner/steve/work/projects/nextflow-cluster"

if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    USER_DIR="${SLURM_SUBMIT_DIR}"
else
    USER_DIR="$(pwd)"
fi
cd "${USER_DIR}" || exit 1

# Parse optional -c flag for custom config path
CONFIG_FILE=""
while getopts "c:" opt 2>/dev/null; do
    case $opt in
        c) CONFIG_FILE="$OPTARG" ;;
    esac
done

# Require -c flag — no default config
if [ -z "${CONFIG_FILE}" ]; then
    echo "ERROR: No config file specified."
    echo "Usage: sbatch ${INSTALL_DIR}/run.sh -c <your_config.config>"
    echo ""
    echo "To get started, create a config from the template:"
    echo "  cp ${INSTALL_DIR}/finkbeiner.config.template ./finkbeiner.config"
    exit 1
fi

# Resolve relative paths against USER_DIR
if [[ "${CONFIG_FILE}" != /* ]]; then
    CONFIG_FILE="${USER_DIR}/${CONFIG_FILE}"
fi

# Extract experiment name from config for the log filename
EXP_NAME=$(grep -m1 "params.experiment" "${CONFIG_FILE}" | sed "s/.*=\s*['\"]//;s/['\"].*//" 2>/dev/null)
EXP_NAME="${EXP_NAME:-unknown}"

# Redirect all output to a user-friendly log file
LOG_FILE="${USER_DIR}/$(whoami)_$(date +%Y%m%d_%H%M%S)_${SLURM_JOB_ID:-local}_${EXP_NAME}.out"
exec > "${LOG_FILE}" 2>&1
echo "Log file: ${LOG_FILE}"

# Disable color output in Nextflow
export NXF_CLI_COLOR=false

# Log environment info
echo "================================="
echo "Starting job on node: $(hostname)"
echo "Install directory: ${INSTALL_DIR}"
echo "User directory:    ${USER_DIR}"
echo "Config file:       ${CONFIG_FILE}"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No GPU detected, running on CPU"
fi
echo "================================="

# Add /usr/bin to PATH for Singularity/Apptainer
export PATH=/usr/bin:$PATH

VALIDATE_SCRIPT="${INSTALL_DIR}/bin/validate_config.py"
CONTAINER="${INSTALL_DIR}/nextflow-cluster.sif"

# Validate config file before running Nextflow
echo "================================="
echo "Validating config file: ${CONFIG_FILE}"
echo "================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    echo ""
    echo "To get started, copy the template config to your directory:"
    echo "  cp ${INSTALL_DIR}/finkbeiner.config.template ${USER_DIR}/finkbeiner.config"
    exit 1
fi

# Check if container exists
if [ ! -f "${CONTAINER}" ]; then
    echo "ERROR: Container not found: ${CONTAINER}"
    exit 1
fi

# Validate config
if python3 "${VALIDATE_SCRIPT}" "${CONFIG_FILE}"; then
    echo "Config validation passed"
    echo "================================="
else
    echo ""
    echo "ERROR: Config validation failed!"
    echo "Please fix the issues in ${CONFIG_FILE} before running the pipeline."
    echo "You can run the validation manually with:"
    echo "  python3 ${VALIDATE_SCRIPT} ${CONFIG_FILE}"
    echo "================================="
    exit 1
fi

# Run Nextflow — pipeline.nf is in INSTALL_DIR, config + all output in USER_DIR
nextflow run "${INSTALL_DIR}/pipeline.nf" \
  -with-apptainer "${CONTAINER}" \
  -c "${CONFIG_FILE}" \
  -work-dir "${USER_DIR}/work_${SLURM_JOB_ID:-$$}" \
  --process.echo true \
  -ansi-log false

NF_EXIT=$?
if [ $NF_EXIT -ne 0 ]; then
    echo ""
    echo "================================="
    echo "Nextflow failed (exit $NF_EXIT). Last 100 lines of .nextflow.log:"
    echo "================================="
    if [ -f "${USER_DIR}/.nextflow.log" ]; then
        tail -100 "${USER_DIR}/.nextflow.log"
    else
        echo "(No .nextflow.log found)"
    fi
    echo "================================="
    exit $NF_EXIT
fi
echo "Nextflow pipeline completed"

