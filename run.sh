#!/bin/bash

# Create a directory for Slurm output logs
mkdir -p /gladstone/finkbeiner/steve/work/projects/nextflow-cluster/slurm-logs

# Slurm job options
#SBATCH --job-name=nextflow-run
#SBATCH --time=08:00:00
#SBATCH -N 1
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

# Get the directory where this script is located
# This works whether the script is run directly or via sbatch
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/finkbeiner.config"
VALIDATE_SCRIPT="${SCRIPT_DIR}/bin/validate_config.py"

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
  -ansi-log false \
  
echo "Nextflow pipeline completed"

