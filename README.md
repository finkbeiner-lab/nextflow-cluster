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

**Development: In the branch called normalization, background subtraction is set up as a separate step. First, this module runs which saves background subtracted images and updates the database with their path. Second, the modules read the database to use background subtracted images rather than raw images, calculating the background, and running the rest of the module. This change was made because there was demand to see the result of background subtraction.**

This module saves normalized images to the analysis directory. It takes shared variables as input including experiment, wells, timepoint, channels, and the normalization method. 

The background subtraction accumulates the tiles in the well per tile per channel and takes the median of each pixel creating a background image. Assuming that cells are more often than not in different places, this operation should just show the background. The background then may be subtracted from the original tiles. 

### Segmentation

Segmentation uses histogram and entropy based thresholding from scikit-image or the method we call sd-from-mean. SD-from-mean calculates the mean of the image, adds the standard deviation times a scale factor, generally 3.5, and uses that as a global threshold. There is also a lower and upper area threshold. This module uses threading. 

### Cellpose Segmentation

[Cellpose](https://www.nature.com/articles/s41592-020-01018-x) is a machine learning model that segments cells. It is based on U-Net and predicts whether the object is a cell or not based on image gradients. 

Cellpose accepts the raw image. I generally leave the default parameters for cellpose, but I change the model. We have cyto, cyto2, and nuclei, all trained on different datasets. 

The cellpose parameters include cell diameter which is generally between 30 and 50. Cellpose was trained on cells with cell diameter 30. The cell probability threshold mean all pixels with value above threshold kept for masks, decrease to find more and larger masks. The flow threshold refers to the gradient and that means all cells with errors below flow threshold are kept. Cellpose uses the gpu if available from the line:

    model = models.Cellpose(gpu=True, model_type=self.opt.model_type)

### Puncta Segmentation

Puncta segmentation is based on difference of gaussians, the same method as fiji. This method blurs the image with two gaussians, one stronger than the other, and subtracts. The result is then thresholded and small peaks (puncta) survive. It helps to run a single tile or well to make sure the parameters are correct. The most sensitive parameters are `sigma1` and `sigma2` which refers to the spread of the gaussian. Default for the sigmas are 2 and 4. 

### Tracking

Tracking is set up as a [minimum cost flow problem](https://en.wikipedia.org/wiki/Minimum-cost_flow_problem). Say we are tracking cells from T0 to T1 in a single tile. In a graph, we have nodes connected by edges. The T0 cells are nodes on the left side of the graph, we're reading left to right. The nodes on the right side of the graph are T1. The left nodes may only connect to one right node. If there are too many left nodes, then some cells vanish. Likewise, if there are too many right nodes, then some cells appear. The edges connecting the nodes are the distance between them. The goal of the minimum cost flow problem is to choose the edges between the nodes that, when you sum them all up, the sum is minimized. 

The minimum cost flow problem is an optimization problem which may be solved with a linear program. The solver used is [Gurobi](https://www.gurobi.com/) which has a free academic license. I used to use GLPK, but the solver was slow and did not permit threading. The license for Gurobi lives in /opt/gurobi. IT has communicated with gurobi and can acquire more free licenses for the cluster if needed. On your work station, your gladstone email suffices to acquire a license. 

If there is an error, run the script [tracking.py](bin/tracking.py) from VS Code while ssh'ed into the docker container and run with `DEBUG=True`. 

### Intensity

This script, [intensity.py](bin/intensity.py), takes the masks of the morphology channel and projects them onto the images from other channels. Then it calculates the max, mean, and min intensity of the region. 

In nextflow, modules.nf, the INTENSITY process uses `each` instead of `val`, which instructs nextflow to run each values from the input list simultaneous. For this, every channel input into intensity.py runs simultaneously. 

### Crop

This module generates image crops used in cnn.py. You can select crop size and which channels to crop. The location of each crop is stored in the database. 

### Montage

This module montages wells. You can choose whether to montage the raw image, a background subtracted image, masks, or tracked masks. There is an option to use the current montage pattern which is like a book (the numbers represent tile numbers)

||||
|---|---|---|
| 1 | 2 | 3 |
| 4 | 5 | 6 |
| 7 | 8 | 9 |

Or `legacy montage` which is like a snake, mostly used for Robo4. 

||||
|---|---|---|
| 3 | 2 | 1 |
| 4 | 5 | 6 |
| 9 | 8 | 7 |


### Plate Montage

Plate montage creates a montage of the whole plate. There is an option for `img_size` so you can reduce the size of each well montage to your chosen pixel length and `norm_intensity` which divides the 16-bit image by the value `norm_intensity` (could be around 2000 to 10000) and multiples by 255 to convert the image to 8-bit. 

### CNN

The [CNN](bin/cnn.py) has multiple architectures, you can choose from a CNN, CNN with batch norm, CNN with dropout, or Resnet. 

The CNN reads the labels from the database. You can choose to train on `celltype`, `name` which refers to the name of the dosage in the dosagedata table, or `stimulate`, which means if the cell was stimulated by the DMD in the Thinking Microscope.

The `label_name` parameter allows you to choose what kind of dosage was added, such as `treatment`, `antibody`, `inhibitor`, or another name entered into the `kind` column of dosagedata. 

The `classes` parameter is a comma separated list of classes for the user to specify the classes to classify. If this is blank, all classes are used. 

The `filters` parameter was added to further filter the dataframe with a key value argument. As an example, `['name', 'cry2mscarlet']` filters the name from the dosagedata table to only include the tag cry2mscarlet. 

The parameter `chosen_channels_for_cnn` is a comma separated list of channels to train on. Leave blank for all channels. 

The argument `num_channels` requires you to specify how many channels you want to train on. 

If you want to train on all samples, set `n_samples=0`, otherwise, set this argument to a number like 100 or 2000. 

The arguments `epochs`, `batch_size`, `learning_rate`, `momentum`, and `optimizer` are common for machine learning. 

*Debugging*: There is a script [cnn_from_dir.py](bin/cnn_from_dir.py) which can be used to train a CNN from a directory rather than from the database. 

### Get CSVS

This module queries the database based on the experiment name and saves each table as a csv. It also generates legacy_celldata.csv, which has column names to match the cell_data.csv from the old version of galaxy. 

To note, [r_merge_demo.R](bin/r_merge_demo.R) shows an example of how to properly merge the generated CSVS. 

## Development: To Do List

### Versioned Analysis

People would like to be able to run different analyses on the same dataset without overwritting previous analysis. As it is, the pipeline overwrites segmentation, intensity, and cell info and only one analysis is stored. I added a version column of type integer to the database to start the process. If researchers could rerun a dataset analysis with new parameters, the pipeline could save the analysis with a new version number, which would be a way to keep track of each analysis. 

### Image Registration

SIFT Registration in Java is very popular. It is set up in Java as [Sift_Registration.java](bin/javamodules/src/main/java/com/finkbeiner/Sift_Registration.java), but not connected to the database. Using JOOQ, I started making calls to the database in [db.java](bin/javamodules/src/main/java/com/finkbeiner/db.java). This file can read rows, get uuids, and update the database. 

### R Modules

There are a few R modules, [survival.R](bin/survival.R) and [gedi.R](bin/gedi.R). The modules run and they use the library RPostgres [rsql.R](bin/rsql.R) to run SQL commands. They are not connected to Nextflow and should be. 

Additionally, [r_merge_demo.R](bin/r_merge_demo.R) shows an example of how to properly merge the generated CSVS. 

### Overlay

The file [overlay.py](bin/overlay.py) runs and needs to be added to [modules.nf](modules.nf) and [pipeline.nf](pipeline.nf). This file overlays the cellid as text over the background subtracted image. 

### Template Maker

Building from the platemap maker gui, the [templatemaker.py](bin/templatemaker.py) is halfway done and would be useful in creating templates to use the roboscopes. 

### Segment Anything Model (SAM)

In addition to Cellpose, SAM should be good at segmenting cells.

## Logs

The logger saves to [finkbeiner_logs](bin/finkbeiner_logs) in the bin directory of this repo. 

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

We use a postgres database. 

- Host: fb-postgres01.gladstone.internal
- User: postgres
- Database: galaxy
- Port: 5432

The password is on Onepassword. 

The schema is currently public and should be migrated to make it easier to see the tables. 

Schemas of the tables are below.

Each table has a column `id` which is a uuid. Columns with the table name + '_id' is a foreign key linking the table's primary key to selected table.

### experimentdata

    id          uuid default uuid_generate_v4() not null
        primary key,
    experiment  varchar
        constraint unique_experiment
            unique,
    researcher  varchar,
    description varchar,
    project     varchar,
    platetype   varchar,
    wellcount   integer,
    imagedir    varchar,
    analysisdir varchar,
    platename   varchar,
    fixed       boolean,
    microscope  varchar

### welldata

This table is a lists wells, celltypes, and the conditions of the cells (control, disease, etc.). 

    id                uuid default uuid_generate_v4() not null
        primary key,
    experimentdata_id uuid                            not null
        references experimentdata
            on delete cascade,
    well              varchar,
    celltype          varchar,
    condition         varchar

### tiledata

This table, tiledata, includes information about the tile, foreign keys to the experiment and well it belongs to, as well as channel. The schema includes time imaged, the paths to the filename (raw image), path to the masks and tracked mask path. 

    id                 uuid default uuid_generate_v4() not null
        primary key,
    experimentdata_id  uuid                            not null
        references experimentdata
            on delete cascade,
    welldata_id        uuid                            not null
        references welldata
            on delete cascade,
    channeldata_id     uuid                            not null
        references channeldata
            on delete cascade,
    tile               integer,
    pid                varchar,
    hours              double precision,
    timepoint          integer,
    overlap            double precision,
    zstep              integer,
    zstep_size         double precision,
    filename           varchar,
    time_imaged        varchar,
    maskpath           varchar,
    trackedmaskpath    varchar,
    segmentationmethod varchar,
    backgroundpath     varchar, 
    registeredpath     varchar

### celldata

Celldata includes foreign keys linking the table to experimentdata, welldata, and tiledata. It includes the centroid coordinates, cellid (tracked), randomcellid (untracked), features about the mask, and whether or not the cell was stimulated by the dmd. 

    id                uuid not null
        constraint celldata_pkey1
            primary key,
    experimentdata_id uuid not null
        references experimentdata
            on delete cascade,
    welldata_id       uuid not null
        references welldata
            on delete cascade,
    tiledata_id       uuid not null
        references tiledata
            on delete cascade,
    cellid            integer,
    randomcellid      integer,
    centroid_x        double precision,
    centroid_y        double precision,
    area              double precision,
    solidity          double precision,
    extent            double precision,
    perimeter         double precision,
    eccentricity      double precision,
    axis_major_length double precision,
    axis_minor_length double precision,
    stimulate         boolean default false

### intensitycelldata

This table, intensitycelldata, includes the intensity information about the cell from celldata. This table is linked to experimentdata, welldata, tiledata, celldata, and channeldata. It includes max, mean, min and std intensity. 

    id                uuid default uuid_generate_v4() not null
        primary key,
    experimentdata_id uuid                            not null
        references experimentdata
            on delete cascade,
    welldata_id       uuid                            not null
        references welldata
            on delete cascade,
    tiledata_id       uuid                            not null
        references tiledata
            on delete cascade,
    celldata_id       uuid                            not null
        references celldata
            on delete cascade,
    channeldata_id    uuid                            not null
        references channeldata
            on delete cascade,
    intensity_max     double precision,
    intensity_mean    double precision,
    intensity_min     double precision,
    intensity_std     double precision

### dosagedata

This table, dosagedata, has information about the platemap. It links to experimentdata and welldata. It shows the name of the dose, kind of dose, and the quantity of the dosage. 

**Todo: needs a units column**. 

    id                uuid default uuid_generate_v4() not null
        primary key,
    experimentdata_id uuid                            not null
        references experimentdata
            on delete cascade,
    welldata_id       uuid                            not null
        references welldata
            on delete cascade,
    name              varchar,
    dosage            double precision,
    kind              varchar

## punctadata

Similar to celldata, this table records puncta. It links to experimentdata, welldata, tiledata, and celldata. There are  columns punctaid (tracked) and randompunctaid (untracked). It includes centroid information and features about the puncta. 

Note tracking puncta is not implemented. 

    id                uuid not null
        primary key,
    experimentdata_id uuid not null
        references experimentdata
            on delete cascade,
    welldata_id       uuid not null
        references welldata
            on delete cascade,
    tiledata_id       uuid not null
        references tiledata
            on delete cascade,
    celldata_id       uuid not null
        references celldata
            on delete cascade,
    punctaid          integer,
    randompunctaid    integer,
    centroid_x        double precision,
    centroid_y        double precision,
    area              double precision,
    solidity          double precision,
    extent            double precision,
    perimeter         double precision,
    eccentricity      double precision,
    axis_major_length double precision,
    axis_minor_length double precision

### intensitypunctadata

Similar to intensitycelldata, this table keeps track of the intensity of puncta. This table links to experimentdata, welldata, tiledata, celldata, punctadata, and channeldata. It records max, min, mean, and std intensity. 

    id                uuid default uuid_generate_v4() not null
        primary key,
    experimentdata_id uuid                            not null
        references experimentdata
            on delete cascade,
    welldata_id       uuid                            not null
        references welldata
            on delete cascade,
    tiledata_id       uuid                            not null
        references tiledata
            on delete cascade,
    celldata_id       uuid                            not null
        references celldata
            on delete cascade,
    punctadata_id     uuid                            not null
        references punctadata
            on delete cascade,
    channeldata_id    uuid                            not null
        references channeldata
            on delete cascade,
    intensity_max     double precision,
    intensity_mean    double precision,
    intensity_min     double precision,
    intensity_std     double precision

## cropdata

Crops enerated from the datastudy repo by the cropping module are stored in this table. This table links to experimentdata, welldata, channeldata, and celldata. The croppaths are used to build a list of files for training, rather than crawling through a directory. 

    id                uuid not null
        primary key,
    experimentdata_id uuid not null
        references experimentdata
            on delete cascade,
    welldata_id       uuid not null
        references welldata
            on delete cascade,
    channeldata_id    uuid not null
        references channeldata
            on delete cascade,
    celldata_id       uuid not null
        references celldata
            on delete cascade,
    croppath          varchar

## modeldata

This table records results from training machine learning models from the CNN module of the datastudy repo. It links to experimentdata. It include the modelname, path to the saved model, path to the wandb file which was saved offline, and stats and parameters about the training. 

    id                uuid not null
        primary key,
    experimentdata_id uuid not null
        references experimentdata
            on delete cascade,
    modelname         varchar,
    modelpath         varchar,
    wandbpath         varchar,
    train_loss        double precision,
    val_loss          double precision,
    train_acc         double precision,
    val_acc           double precision,
    epochs            integer,
    n_samples         integer,
    num_channels      integer,
    learning_rate     double precision,
    batch_size        integer,
    momentum          double precision,
    optimizer         varchar,
    modeltype         varchar

## modelcropdata

This table, modelcropdata, keeps records of the prediction and ground truth of each crop after training with the CNN. This links to modeldata, experimentdata, welldata, and celldata. The column stage refers to training, validation, or testing. The prediction and groundtruth are numeric and the prediction_label and groundtruth_label are, if you're running classification, the corresponding label for the ground prediction and groundtruth. 

    id                uuid not null
        primary key,
    model_id          uuid not null
        references modeldata
            on delete cascade,
    experimentdata_id uuid not null
        references experimentdata
            on delete cascade,
    welldata_id       uuid not null
        references welldata
            on delete cascade,
    celldata_id       uuid not null
        references celldata
            on delete cascade,
    stage             varchar,
    output            double precision,
    prediction        double precision,
    groundtruth       double precision,
    prediction_label  varchar,
    groundtruth_label varchar

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
