# DATASTUDY 
A Nextflow Image Analysis Pipeline for HPC with Slurm and Docker. 

## About
The main nextflow script, pipeline.nf, includes segmentation, tracking, puncta segmentation, survival analysis, and a CNN to analyze image data. 
The modules read and write to and from a postgres database. The modules include python, R, and Java for ImageJ.

## Running the Pipeline

Edit the finkbeiner.config file. 

Then run

    nextflow run pipeline.nf

The file `pipeline.nf` lives in `datastudy/pipeline.nf`. On the cluster, the repo is in 

    /gladstone/finkbeiner/imaging-work/datastudy

### Select Modules

Choose which module you want to run. If you want to run them, set them to true, otherwise false. 

### Shared Variables

These are variables that affect every other module including the experiment, selected wells, timepoint, and channels to include or exclude. The morphology channel. And the `img_norm_name` which is generally `subtraction`, the background subtraction method to normalization images. As of Jan 16, 2024, the branch called `normalization` is for saving background subtracted images as a separate step. Currently, background subtraction is an option in relevation modules. 

### Variables Per Module

The rest of the parameters are arguments for each module. See descriptions of modules below.

## Modules

### Register Experiment
[Register Experiment](bin/register_experiment.py) adds an experiment to the database if not there already. If the experiment needs to be overwritten, there is a flag, `overwrite_experiment`, which you can set to `1` which will allow the experiment to be overwritten. 

Register Experiment requires the path to the raw images (input path), the output path which is your analysis directory (path with GXYTMP) and template files depending on your experiment. If running from the current microscope code, your experiment should already be on the database. If it is not, them use the xlsx template. If you are using the old code, referred to as legacy code, then you have a csv template. There is a template converter option for legacy code called `robo_file`. If you are using the ixm, then `ixm_hts_file` accepts a file of type `.HTS`. The IXM template converter generates an xlsx template, saves it to the analysis directory and uses that to run register experiment, all in one go. If you are using the IXM or legacy code, then you need a platemap. The file [platemap_maker.py](bin/platemap_maker.py) runs a GUI which will generate a platemap.csv for you. It saves to `datastudy/bin`, move it to where you like. The illumination file is specific for the IXM and is used to populate channel information about the IXM. The pickle file `outfile` is deprecated.

Troubleshooting: Unused arguments from nextflow should be single character strings such as 
    
    params.robo_file = '.'  // Legacy Template Path for Roboscopes (csv)

It would be better to change this to passing a null value. 

### Update Paths

If you run the transfer script to move the images from the microscope computer to the NAS, the database may have recorded the local path instead of the new path on the NAS. This module just requires the experiment name. 

### Normalization Visualization

This module saves normalized images to the analysis directory. It takes shared variables as input including experiment, wells, timepoint, channels, and the normalization method. 

### Segmentation



### Cellpose Segmentation

### Puncta Segmentation

### Tracking

### Intensity

### Crop

### Montage

### Plate Montage

### CNN

### Get CSVS

## Docker Image
The docker image may be obtained here:

`docker pull jdlamstein/datastudy`

The Dockerfile to build this image is included in this repo. 

Some standard Docker commands include 

- `docker ps` : See running containers. 
- `docker ps -a` : See all containers including stopped. 
- `docker image list` : See the images downloaded to your computer

To check if Docker is working run

    docker run hello-world

If you are on linux, then after installing docker it will likely require sudo to run. To avoid running as root, follow the [post installation instructions for linux](https://docs.docker.com/engine/install/linux-postinstall/).

## VS Code

[VS Code](https://code.visualstudio.com/) is open source software which allows you to ssh into a running docker container. If you install extensions, you can run all the languages supported in this project's docker image including python, R, and java. This code editor allows you to run single modules, useful for development. 

## Nextflow
To install Nextflow on a new system, follow these [instructions](https://www.nextflow.io/docs/latest/getstarted.html).

The instructions include installing java greater than version 11. 

Install Nextflow, make it executable, and move it to the path. Or append the location of the nextflow to your path. 

If you want to know the location of your nextflow executable run `whereis nextflow`.

## Database

The Image Analysis Pipeline and Roboscopes use a Postgres Database. The credentials are stored in OnePassword. If you need to check the credentials, ask IT. 

## Slurm

Slurm is installed on galaxy.gladstone.internal as the headnode, and fb-docker-compute07, fb-docker-compute08, fb-gpu-compute01 as workers. 

Run sinfo to see a recent list of worker nodes. 

Some useful slurm commands:

* sinfo - get info about slurm
* squeue - see what slurm is running
* sbatch - run an executable file (.sh file)
* srun  - run a command in slurm
* scontrol - used to initialize and set parameters for slurm. Useful when adding and removing nodes or resetting nodes in a down state. 

**EXAMPLE:**

Run this command

    cd /gladstone/finkbeiner/imaging-work
    sbatch sdocker.sh
    sbatch sdocker.sh
    sbatch sdocker.sh
    sbatch sdocker.sh
    sbatch sdocker.sh
    squeue
    sinfo

Squeue should show jobs running on nodes 7 and 8. 
sdocker.sh is a sleep command after starting up docker through bash:

    #!/bin/bash
    sudo docker run --rm -u fbgalaxy --mount type=bind,source="/gladstone/finkbeiner/linsley",target="/gladstone/finkbeiner/linsley" galaxy-docker sleep 60

If you just run the sleep command, only one node runs it. 
Just the sleep command is sbatch.sh in the work directory.

Slurm is managed by the supervisor systemctl. To enable, restart, start, and stop
Note "ctl" is for controller, "d" is for daemon. 

* Controller: systemctl enable slurmctld
* Database: systemctl enable slurmdbd
* Compute Nodes: systemctl enable slurmd

**Add new nodes:**

To add a new node, you'll need to access the slurm.conf file in `/etc/slurm/slurm.conf`.
See documentation:
https://slurm.schedmd.com/faq.html#add_nodes

Slurm Logs are stored in 

    /var/log/slurmctld.log
    /var/log/slurmd.log

If a node is down, you'll likely need to restart slurm on all nodes and restart with scontrol. 
https://slurm.schedmd.com/faq.html#return_to_service
To get the reason it's down, try

`scontrol show node fb-docker-compute07`

If the reason is it's down from a restart run, 

`scontrol update NodeName=whatever State=RESUME"`

If slurm is down, it may be munge. This happened once after a restart. Munge had permission errors. 

I ran: 

`sudo systemctl start munge`

I got: 

    Job for munge.service failed because the control process exited with error code.
    See "systemctl status munge.service" and "journalctl -xe" for details.
    Apr 12 16:20:44 galaxy.gladstone.internal munged[36579]: munged: Error: Failed to check logfile "/var/log/munge/munged.log": Permission denied

Munge should be run as munge user, not root. 
