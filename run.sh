#!/bin/bash

# Create a directory for Slurm output logs
mkdir -p /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/slurm-logs

# Slurm job options
#SBATCH --job-name=nextflow-run
#SBATCH --time=08:00:00
#SBATCH -N 2
#SBATCH --output=/gladstone/finkbeiner/steve/work/projects/nextflow-cluster/slurm-logs/slurm-%j.out
##SBATCH --gres=gpu:v100:1
#SBATCH --distribution=block:block


# Disable color output in Nextflow
export NXF_CLI_COLOR=false

# Log environment info
echo "================================="
echo "Starting job on node: $(hostname)"

if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo "No GPU detected, running on CPU"
fi
echo "================================="

# Add /usr/bin to PATH for Singularity/Apptainer
export PATH=/usr/bin:$PATH

# Use submit directory when running under Slurm (sbatch copies script to spool dir);
# otherwise use the directory where this script lives
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
    WORK_DIR="${SLURM_SUBMIT_DIR}"
else
    WORK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
CONFIG_FILE="${WORK_DIR}/finkbeiner.config"
VALIDATE_SCRIPT="${WORK_DIR}/bin/validate_config.py"

# Validate config file before running Nextflow
echo "================================="
echo "Validating config file: ${CONFIG_FILE}"
echo "================================="

# Check if config file exists
if [ ! -f "${CONFIG_FILE}" ]; then
    echo "ERROR: Config file not found: ${CONFIG_FILE}"
    exit 1
fi

# Check if validation script exists
if [ ! -f "${VALIDATE_SCRIPT}" ]; then
    echo "WARNING: Validation script not found: ${VALIDATE_SCRIPT}"
    echo "Skipping validation and proceeding with Nextflow run..."
    echo "================================="
else
    # Run validation script
    if python3 "${VALIDATE_SCRIPT}" "${CONFIG_FILE}"; then
        echo "✓ Config validation passed"
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
fi

# Run Nextflow inside Apptainer container
nextflow run pipeline.nf \
  -with-apptainer /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/nextflow-cluster.sif \
  -c finkbeiner.config \
  --process.echo true \
  -ansi-log false

NF_EXIT=$?
if [ $NF_EXIT -ne 0 ]; then
    echo ""
    echo "================================="
    echo "Nextflow failed (exit $NF_EXIT). Last 100 lines of .nextflow.log:"
    echo "================================="
    if [ -f "${WORK_DIR}/.nextflow.log" ]; then
        tail -100 "${WORK_DIR}/.nextflow.log"
    else
        echo "(No .nextflow.log found at ${WORK_DIR}/.nextflow.log)"
    fi
    echo "================================="
    exit $NF_EXIT
fi
echo "Nextflow pipeline completed"

