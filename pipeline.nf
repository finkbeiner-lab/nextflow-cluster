#!/usr/bin/env nextflow
/*
 * pipeline input parameters
 https://nextflow-io.github.io/patterns/conditional-process/
 */
// SHARED VARIABLES
params.wells_toggle = 'include' // ['include', 'exclude']
params.chosen_wells = 'all'  // 'A1,A2,A7', or 'A1-A6' or 'B07,G06' or 'A1' or 'all'

params.timepoints_toggle = 'include' // ['include', 'exclude']
params.chosen_timepoints = 'all'  // 'T0', 'T0-T7', or 'all'

params.channels_toggle = 'include' // ['include', 'exclude']
params.chosen_channels = ''  // 'RFP1', 'RFP1,GFP,DAPI', 'all'

params.experiment = '20230928-MsNeu-RGEDItau1'  // Experiment name 
params.morphology_channel = 'Confocal-GFP16'  // Your morphology channel
params.analysis_version = 1  // Analysis version. Change if you're rerunning analysis and want to save previous iteration.
params.img_norm_name = 'subtraction' // ['identity', 'subtraction', 'division']

// SELECT MODULES
params.DO_UPDATEPATHS = false
params.DO_REGISTER_EXPERIMENT = false
params.DO_SEGMENTATION = false
params.DO_CELLPOSE_SEGMENTATION = false
params.DO_TRACKING = false
params.DO_INTENSITY = false
params.DO_CROP = false
params.DO_MONTAGE = true
params.DO_PLATEMONTAGE = false
params.DO_CNN = false
params.DO_GET_CSVS = false

// Variables per module

// REGISTER_EXPERIMENT
params.input_path = ''  // path to raw images
params.output_path = ''  // analysis directory
params.template_path = ''  // xlsx template
params.ixm_hts_file = ''  // IXM HTS Template
params.platemap_path = '' // Platemap path (csv)
params.illumination_file = '/gladstone/finkbeiner/robodata/IXM Documents/illumination-setting-2023-06-16.ILS'  // Path to IXM Illumination file. On metaxpres -> Control -> Devices -> Configure Illumination -> Backup
params.robo_num = 0  // [0,3,4]
params.chosen_channels_for_register_exp = 'all'  // Used by Montage as well

// SEGMENTATION
params.segmentation_method = 'sd_from_mean' // ['sd_from_mean', 'triangle', 'minimum', 'yen']
params.lower_area_thresh = 50
params.upper_area_thresh = 36000
params.sd_scale_factor = 3.5

// CELLPOSE SEGMENTATION
params.model_type = 'cyto2' // ['cyto', 'nuclei', 'cyto2']
params.batch_size = 16
params.cell_diameter = 50  // default is 30 pixels
params.flow_threshold = 0.4
params.cell_probability = 0.0

// TRACKING
params.distance_threshold = 300 // distance that a cell must be new
params.voronoi_bool = true // distance that a cell must be new

// INTENSITY
params.target_channel = ['Epi-RFP16']  // List of channels. Run in parallel.

// CROP
params.crop_size = 300
params.target_channel_crop = ['Confocal-GFP16', 'Epi-RFP16']  // List of channels. Run in parallel.

// MONTAGE and PLATEMONTAGE
params.tiletype = 'filename'  // ['filename', 'maskpath', 'trackedmaskpath']
params.montage_pattern = 'standard'  // ['standard', 'legacy']
params.well_size_for_platemontage = 300  // side length for well
params.norm_intensity = 2000 // normalization intensity for well  (img / norm_intensity)  * 255

// CNN
params.label_type = 'stimulate'  // 'celltype', 'name', 'stimulate'
params.label_name = null // Match the kind of dosage added. Treatment, Antibody, Inhibitor, etc
params.classes = null // Comma separated list of classes. If all classes in experiment, leave blank.
params.img_norn_name_cnn = 'identity' // identity, subtraction, division
params.filters = 'name,cry2mscarlet' // oolumnname,value used to filter down datasets
params.chosen_channels_for_cnn = 'RFP1,RFP2'  // blank for all. Select which channels will be included in input
params.num_channels = 2  // number of channels to include in input model_type
params.n_samples = 100  // 0 for all. Otherwise set a number.model_type
params.epochs = 1  // number of epochs. Number of times the model will the see the entire dataset.
params.batch_size = 16  // Number of images the model sees simulataneously. 16, 32, 64 are good numbers. 128 is good too, gpu size permitting.
params.learning_rate = 1e-4
params.momentum = 0.9  // Only for Stochastic Gradient Descent (SGD)
params.optimizer = 'adam' // 'adam', 'sgd'


////////////////////////////////////////////////////////////////////////////////////////////////////////////////

input_path_ch = Channel.of(params.input_path)
output_path_ch = Channel.of(params.output_path)
template_path_ch = Channel.of(params.template_path)
ixm_hts_file_ch = Channel.of(params.ixm_hts_file)
platemap_path_ch = Channel.of(params.platemap_path)
illumination_file_ch = Channel.of(params.illumination_file)
robo_num_ch = Channel.of(params.robo_num)
chosen_channels_for_register_exp_ch = Channel.of(params.chosen_channels_for_register_exp)
experiment_ch = Channel.of(params.experiment)
seg_ch = Channel.of(params.segmentation_method)
norm_ch = Channel.of(params.img_norm_name)
lower_ch = Channel.of(params.lower_area_thresh)
upper_ch = Channel.of(params.upper_area_thresh)
sd_ch = Channel.of(params.sd_scale_factor)
well_ch = Channel.of(params.chosen_wells)
tp_ch = Channel.of(params.chosen_timepoints)
well_toggle_ch = Channel.of(params.wells_toggle)
tp_toggle_ch = Channel.of(params.timepoints_toggle)
channel_toggle_ch = Channel.of(params.channels_toggle)
target_channel_ch = Channel.from(params.target_channel)
target_channel_crop_ch = Channel.from(params.target_channel_crop)
morphology_ch = Channel.of(params.morphology_channel)
distance_threshold_ch = Channel.of(params.distance_threshold)
voronoi_bool_ch = Channel.of(params.voronoi_bool)
crop_size_ch = Channel.of(params.crop_size)
tiletype_ch = Channel.of(params.tiletype)
montage_pattern_ch = Channel.of(params.montage_pattern)
well_size_for_platemontage_ch = Channel.of(params.well_size_for_platemontage)
norm_intensity_ch = Channel.of(params.norm_intensity)

model_type_ch = Channel.of(params.model_type)
batch_size_ch = Channel.of(params.batch_size)
cell_diameter_ch = Channel.of(params.cell_diameter)
flow_threshold_ch = Channel.of(params.flow_threshold)
cell_probability_ch = Channel.of(params.cell_probability)

label_type_ch = Channel.of(params.label_type)
label_name_ch = Channel.of(params.label_name)
classes_ch = Channel.of(params.classes)
img_norn_name_cnn_ch = Channel.of(params.img_norn_name_cnn)
filters_ch = Channel.of(params.filters)
chosen_channels_for_cnn_ch = Channel.of(params.chosen_channels_for_cnn)
num_channels_ch = Channel.of(params.num_channels)
n_samples_ch = Channel.of(params.n_samples)
epochs_ch = Channel.of(params.epochs)
batch_size_ch = Channel.of(params.batch_size)
learning_rate_ch = Channel.of(params.learning_rate)
momentum_ch = Channel.of(params.momentum)
optimizer_ch = Channel.of(params.optimizer)

include { REGISTER_EXPERIMENT; SEGMENTATION;
    CELLPOSE; PUNCTA; TRACKING; INTENSITY;
    CROP; MONTAGE; PLATEMONTAGE; CNN; GETCSVS; MULT; BASHEX;UPDATEPATHS;
    SPLITLETTERS; CONVERTTOUPPER } from './modules.nf'

params.outdir = 'results'

log.info """\
    ANALYSIS DATASTUDY PIPELINE!
    ===================================
    experiment: ${params.experiment}
    wells: ${params.chosen_wells}
    timepoints: ${params.chosen_timepoints}
    channels: ${params.chosen_channels}
    Register Experiment: ${params.DO_REGISTER_EXPERIMENT}
    Update Database Paths: ${params.DO_UPDATEPATHS}
    Segmentation: ${params.DO_SEGMENTATION}
    Tracking: ${params.DO_TRACKING}
    Intensity: ${params.DO_INTENSITY}
    Crop: ${params.DO_CROP}
    Montage: ${params.DO_MONTAGE}
    Plate Montage: ${params.DO_PLATEMONTAGE}
    CNN: ${params.DO_CNN}
    Get CSVS: ${params.DO_GET_CSVS}
    """
    .stripIndent()

workflow {
    if (params.DO_UPDATEPATHS && !params.DO_REGISTER_EXPERIMENT){
        updatepath_ch = UPDATEPATHS(experiment_ch)
        updatepath_ch.view { it }
        updatepaths_result= UPDATEPATHS.out
    }
    else{
        updatepaths_result = true
    }
    if (params.DO_REGISTER_EXPERIMENT) {
        REGISTER_EXPERIMENT(input_path_ch, output_path_ch, template_path_ch, platemap_path_ch, ixm_hts_file_ch, robo_num_ch,
        well_ch, tp_ch,chosen_channels_for_register_exp_ch, well_toggle_ch, tp_toggle_ch, channels_toggle)
        register_result = REGISTER_EXPERIMENT.out
    }
    else {
        register_result = true
    }
    if (params.DO_SEGMENTATION) {
        seg_ch = SEGMENTATION(register_result, experiment_ch, morphology_ch, seg_ch, norm_ch, lower_ch, upper_ch, sd_ch,
        well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
        seg_ch.view { it }
        seg_result = SEGMENTATION.out
    }
    else {
        seg_result = true
    }
    if (params.DO_CELLPOSE_SEGMENTATION) {

        cellpose_ch = CELLPOSE(register_result, experiment_ch, batch_size_ch, cell_diameter_ch, flow_threshold_ch, 
        cell_probability_ch, model_type_ch, morphology_ch, seg_ch, lower_ch, upper_ch, sd_ch,
        well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
        cellpose_ch.view { it }
        cellpose_result = CELLPOSE.out
    }
    else {
        cellpose_result = true
    }

    if (params.DO_TRACKING) {
        track_ch = TRACKING(seg_result, experiment_ch, distance_threshold_ch, voronoi_bool_ch, well_ch, tp_ch, morphology_ch,
        well_toggle_ch, tp_toggle_ch)
        track_result = TRACKING.out
    }
    else {
        track_result = true
    }
    if (params.DO_MONTAGE) {

        montage_ch = MONTAGE(track_result, experiment_ch, tiletype_ch, montage_pattern_ch, well_ch, tp_ch, chosen_channels_for_register_exp_ch,
        well_toggle_ch, tp_toggle_ch, channel_toggle_ch)
        montage_result = MONTAGE.out
    }
    if (params.DO_PLATEMONTAGE) {

        platemontage_ch = PLATEMONTAGE(track_result, experiment_ch, well_size_for_platemontage_ch, norm_intensity_ch, tiletype_ch, montage_pattern_ch, well_ch, tp_ch, chosen_channels_for_register_exp_ch,
        well_toggle_ch, tp_toggle_ch, channel_toggle_ch)
        montage_result = PLATEMONTAGE.out
    }
    else {
        montage_result = true
    }
    if (params.DO_INTENSITY) {
        int_ch = INTENSITY(track_result, experiment_ch, norm_ch, morphology_ch, target_channel_ch, well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
        intensity_result = INTENSITY.out
        int_ch.view { it }
    }
    else {
        intensity_result = true
    }
    if (params.DO_CROP) {
        crop_ch = CROP(track_result, experiment_ch, target_channel_crop_ch, morphology_ch, crop_size_ch, well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
        crop_ch.view { it }
        crop_result = CROP.out
    }
    else {
        crop_result = true
    }
    if (params.DO_CNN) {
        cnn_ch = CNN(crop_result, experiment_ch, label_type_ch, label_name_ch, classes_ch, img_norn_name_cnn_ch, filters_ch, num_channels_ch, n_samples_ch,
        epochs_ch, batch_size_ch, learning_rate_ch, momentum_ch, optimizer_ch, well_ch, tp_ch, chosen_channels_for_cnn_ch,
        well_toggle_ch, tp_toggle_ch, channel_toggle_ch)
        cnn_result = CNN.out
    }
    else {
        cnn_result = true
    }
    //     MULT(register_result, seg_result, track_result, intensity_result, crop_result, cnn_result)
    csv_ready = collect(register_result, seg_result, track_result, intensity_result, crop_result, cnn_result)
    //     csv_channel = Channel
    //         .of(register_result, seg_result, track_result, intensity_result, crop_result, cnn_result )
    //         .max()
    //         .view { "Max value is $it" }
    csv_ready = true
    if (params.DO_GET_CSVS) {
        csv_ch = GETCSVS(csv_ready, experiment_ch)
        csv_ch.view { it }
    }
}

workflow.onComplete {
    log.info(workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : 'Oops .. something went wrong')
}
