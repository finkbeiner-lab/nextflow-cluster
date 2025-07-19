#!/bin/bash

# General Information
# Lines starting with '#SBATCH' are Slurm options. Lines with two '##' are disabled.
# To enable an option remove the second '#', or to disable it add a second '#'
#
# If you have any questions using Slurm, please contact the IT Service Desk:
# Web: http://help.gladstone.org
# Email: it-servicedesk@gladstone.ucsf.edu
# Phone: (415) 734-2500

# Available Resources
# If your job isn't running, it may be because there may not be any available resources. Finding what is available below:
# CPU & RAM: You can run 'scontrol show nodes' to see the details of all compute nodes.
#            You can also look at a specific node by adding the hostname, such as 'scontrol show nodes fb-gpu-compute01.gladstone.internal'
#            For each server, CfgTRES shows total number of CPUs and RAM for each server. AllocTRES shows the amount of CPUs and RAM that's been allocated.
#
# GPUs: GPUs in use can be found by running 'squeue -h -t R -O gres | grep gpu|wc -l'


# create directory for Slurm output
mkdir -p /gladstone/finkbeiner/steve/work/projects/datastudy/slurm-logs

# General Options
#SBATCH --job-name=testjob # friendly name for the slurm job. Shown in job info and 'squeue'.
#SBATCH --time=0-08:00:00  # Real time for the job in the formt of D-HH:MM:SS.
#SBATCH -N 1               # Number of physical servers requested. Useful for fater inter-GPU communication.
#SBATCH -p interactive            # Partition (Queue) to submit to. Defaults to 'galaxy' unless 'interactive' is defined. This allows interactive jobs to take priority over automatic Galaxy jobs.
#SBATCH --output /gladstone/finkbeiner/steve/work/projects/datastudy/slurm-logs/slurm-%j.out


# CPU Options
#SBATCH --cpus-per-task 4  # CPUs requested per GPU.
#SBATCH --mem=64G  # Memory for the job.

# GPU Options
#SBATCH --gpus=v100:1           # Type and number of GPUs requested. Available options are A100 and h100, and 1-4 for A100s or 1-8 for H100s.
##SBATCH --cpus-per-gpu=16 # CPUs per GPU requested. Disable '--cpus' above if using this option.
##SBATCH --mem-per-gpu=12G   # Memory per GPU requested. Disable '--mem" above if using this option.
#SBATCH --gres=gpu:v100:1
#SBATCH --constraint="v100"
##SBATCH --nodelist=fb-gpu-compute01,fb-gpu-compute02

#SBATCH --nodelist=fb-docker-compose01

# Set the environment variable to disable color output
export NXF_CLI_COLOR=false



# Run your code below
echo -e "\n================================="

echo -e "\nStarting job on the Slurm node..."

echo -e "\nRunning on node: $(hostname)"


if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
else
    echo -e "\nNo GPU detected, running on CPU"
fi

echo -e "\n=================================="

# Add /usr/bin to PATH explicitly for singularity
export PATH=/usr/bin:$PATH


#nextflow run hello.nf -with-singularity /gladstone/finkbeiner/steve/work/projects/datastudy/datastudy-cluster.sif -ansi-log false
#nextflow run pipeline.nf -with-singularity /gladstone/finkbeiner/steve/work/projects/datastudy/datastudy-cluster.sif -ansi-log false -c finkbeiner.config 

# Run the Singularity container with Nextflow
apptainer exec \
    --bind $HOME/datastudy/bin:/app \
    --bind $HOME/opt/gurobi:/opt/gurobi \
    --bind /gladstone/finkbeiner:/gladstone/finkbeiner:rw \
    --nv \
    /gladstone/finkbeiner/steve/work/projects/datastudy/nextflow-cluster.sif \
    nextflow run gpu.nf -c finkbeiner.config -ansi-log false

echo -e "\nJob Completed on the Slurm node..."
