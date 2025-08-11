#!/bin/bash

# Create a directory for Slurm output logs
mkdir -p /gladstone/finkbeiner/steve/work/projects/datastudy/slurm-logs

# Slurm job options
#SBATCH --job-name=testjob
#SBATCH --time=08:00:00
#SBATCH -N 1
#SBATCH --output=/gladstone/finkbeiner/steve/work/projects/datastudy/slurm-logs/slurm-%j.out
##SBATCH --gres=gpu:v100:1

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

# Run Nextflow inside Apptainer container
nextflow run pipeline.nf \
  -with-apptainer /gladstone/finkbeiner/steve/work/projects/datastudy/nextflow-cluster.sif \
  -c finkbeiner.config \
  --process.echo true \
  -ansi-log false \
  
echo "Nextflow pipeline completed"

