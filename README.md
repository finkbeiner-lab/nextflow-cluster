# DATASTUDY #
A Nextflow Image Analysis Pipeline for HPC with Slurm and Docker. 

## About ##
The main nextflow script, pipeline.nf, includes segmentation, tracking, puncta segmentation, survival analysis, and a CNN to analyze image data. 
The modules read and write to and from a postgres database. The modules include python, R, and Java for ImageJ.

## Docker Image ##
The docker image may be obtained here:
docker pull jdlamstein/datastudy
The Dockerfile to build this image is included in this repo. 

## Cluster ##
The nextflow executable lives in /usr/local/bin.