#!/usr/bin/env nextflow


input_path_ch = Channel.of(params.input_path)
output_path_ch = Channel.of(params.output_path)
template_path_ch = Channel.of(params.template_path)
ixm_hts_file_ch = Channel.of(params.ixm_hts_file)
robo_file_ch = Channel.of(params.robo_file)
platemap_path_ch = Channel.of(params.platemap_path)
illumination_file_ch = Channel.of(params.illumination_file)
robo_num_ch = Channel.of(params.robo_num)
chosen_channels_for_register_exp_ch = Channel.of(params.chosen_channels_for_register_exp)
experiment_ch = Channel.of(params.experiment)
overwrite_experiment_ch = Channel.of(params.overwrite_experiment)
seg_ch = Channel.of(params.segmentation_method)
puncta_seg_ch = Channel.of(params.puncta_segmentation_method)
puncta_manual_thresh_ch = Channel.of(params.puncta_manual_thresh)
sigma1_ch = Channel.of(params.sigma1)
sigma2_ch = Channel.of(params.sigma2)
norm_ch = Channel.of(params.img_norm_name)
lower_ch = Channel.of(params.lower_area_thresh)
upper_ch = Channel.of(params.upper_area_thresh)
sd_ch = Channel.of(params.sd_scale_factor)
// well_ch is defined later in the workflow
tile_ch = Channel.of(params.tile)
use_aligned_tiles_ch = Channel.of(params.use_aligned_tiles)
tp_ch = Channel.of(params.chosen_timepoints)
channel_ch = Channel.of(params.chosen_channels)
well_toggle_ch = Channel.of(params.wells_toggle)
tp_toggle_ch = Channel.of(params.timepoints_toggle)
channel_toggle_ch = Channel.of(params.channels_toggle)
image_overlap_ch = Channel.of(params.image_overlap) //austin added "_ch" 5/17
target_channel_crop_ch = Channel.from(params.target_channel_crop)
puncta_target_channel_ch = Channel.from(params.puncta_target_channel)
morphology_ch = Channel.of(params.morphology_channel)
distance_threshold_ch = Channel.of(params.distance_threshold)
voronoi_bool_ch = Channel.of(params.voronoi_bool)
track_type_ch           = Channel.of(params.track_type)
motion_ch               = Channel.of(params.motion)
crop_size_ch = Channel.of(params.crop_size)

// Define target_channel_ch globally so it's available to all workflows
def target_channel_str = params.target_channel instanceof List 
    ? params.target_channel.join(',')
    : params.target_channel
target_channel_ch = Channel.value(target_channel_str)
// Fallback if tiletype is missing from config
if (!params.tiletype) {
    params.tiletype = 'maskpath'
}

tiletype_ch = Channel.of(params.tiletype)
montage_pattern_ch = Channel.of(params.montage_pattern)
well_size_for_platemontage_ch = Channel.of(params.well_size_for_platemontage)
norm_intensity_ch = Channel.of(params.norm_intensity)
shift_ch = Channel.of(params.shift)//KS edit for overlay
contrast_ch = Channel.of(params.contrast)//KS edit for overlay


aligntiletype_ch = Channel.of(params.aligntiletype)
alignment_algoritm_ch = Channel.of(params.alignment_algorithm)
dir_structure_ch = Channel.of(params.dir_structure)
imaging_mode_ch = Channel.of(params.imaging_mode)
shift_dict_ch = Channel.of(params.shift_dict)



model_type_ch = Channel.of(params.model_type)
batch_size_cellpose_ch = Channel.of(params.batch_size_cellpose)
cell_diameter_ch = Channel.of(params.cell_diameter)
flow_threshold_ch = Channel.of(params.flow_threshold)
cell_probability_ch = Channel.of(params.cell_probability)

cnn_model_type_ch = Channel.of(params.cnn_model_type)
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






include { OVERLAY;REGISTER_EXPERIMENT;ALIGN_TILES_DFT;ALIGN_MONTAGE_DFT;SEGMENTATION;SEGMENTATION_MONTAGE;
    CELLPOSE; PUNCTA; TRACKING; TRACKING_MONTAGE; ALIGNMENT; INTENSITY; 
    CROP; CROP_MASK; MONTAGE; PLATEMONTAGE; CNN; GETCSVS; BASHEX; UPDATEPATHS; NORMALIZATION; COPY_MASK_TO_TRACKED; OVERLAY_MONTAGE; BUNDLED_WORKFLOW_IXM; BUNDLED_STD_WORKFLOW} from './modules.nf'

params.outdir = 'results'

log.info """\
    ANALYSIS DATASTUDY PIPELINE!
    ===================================
    experiment: ${params.experiment}
    wells: ${params.chosen_wells}
    timepoints: ${params.chosen_timepoints}
    channels: ${params.chosen_channels}
    tiletype: ${params.tiletype}
    Register Experiment: ${params.DO_REGISTER_EXPERIMENT}
    Update Database Paths: ${params.DO_UPDATEPATHS}
    Copy Mask to Masktracked: ${params.DO_COPY_MASK_TO_TRACKED}
    Segmentation: ${params.DO_SEGMENTATION}
    Cellpose: ${params.DO_CELLPOSE_SEGMENTATION}
    View Normalization: ${params.DO_VIEW_NORMALIZATION_IMAGES}
    Tracking: ${params.DO_TRACKING}
    Intensity: ${params.DO_INTENSITY}
    Crop: ${params.DO_CROP}
    Mask Crop: ${params.DO_MASK_CROP}
    Montage: ${params.DO_MONTAGE}
    Align Tiles: ${params.DO_ALIGN_TILES_DFT}
    Align Montage: ${params.DO_ALIGN_MONTAGE_DFT}
    Segmentation Montage: ${params.DO_SEGMENTATION_MONTAGE}
    Tracking Montage: ${params.DO_TRACKING_MONTAGE}
    Plate Montage: ${params.DO_PLATEMONTAGE}
    CNN: ${params.DO_CNN}
    Get CSVS: ${params.DO_GET_CSVS}
    Overlay: ${params.DO_OVERLAY}
    Overlay Montage: ${params.DO_OVERLAY_MONTAGE}
    Standard Workflow: ${params.DO_STD_WORKFLOW}
    Standard IXM Workflow: ${params.DO_STD_WORKFLOW_IXM}
    Bundled Standard Workflow: ${params.DO_BUNDLED_STD_WORKFLOW}
    """
    .stripIndent()

workflow {

    if (params.DO_UPDATEPATHS && !params.DO_REGISTER_EXPERIMENT){
        updatepath_ch = UPDATEPATHS(experiment_ch)
        updatepath_ch.view { it }
        updatepaths_result= UPDATEPATHS.out
    }
    else{
        updatepaths_result = Channel.of(true)
    }

    if (params.DO_COPY_MASK_TO_TRACKED) {
        copy_masktracked_ch = COPY_MASK_TO_TRACKED(
            updatepaths_result,
            experiment_ch,
            well_ch,
            tp_ch,
            well_toggle_ch,
            tp_toggle_ch,
            tile_ch
        )
        copy_masktracked_ch.view { it }
        copy_masktracked_result = COPY_MASK_TO_TRACKED.out
    } else {
        copy_masktracked_result = Channel.of(true)
    }


    if (params.DO_REGISTER_EXPERIMENT) {
        REGISTER_EXPERIMENT(input_path_ch, output_path_ch, template_path_ch, platemap_path_ch, ixm_hts_file_ch, robo_file_ch, 
        overwrite_experiment_ch, robo_num_ch, illumination_file_ch,
        well_ch, tp_ch, chosen_channels_for_register_exp_ch, well_toggle_ch, tp_toggle_ch, channel_toggle_ch)
        register_result = REGISTER_EXPERIMENT.out
    }
    else {
        register_result = Channel.of(true)
    }

    if (params.DO_VIEW_NORMALIZATION_IMAGES) {
        norm_flag = updatepaths_result.mix(register_result).collect()
        norm_view_ch = NORMALIZATION(norm_flag, experiment_ch, norm_ch, 
        well_ch, channel_ch, tp_ch, 
        well_toggle_ch, channel_toggle_ch, tp_toggle_ch)
        norm_view_ch.view {it}
        norm_result = NORMALIZATION.out
    }

    if (params.DO_SEGMENTATION) {
        register_flag = updatepaths_result.mix(register_result).collect()
        seg_ch = SEGMENTATION(register_flag, experiment_ch, morphology_ch, seg_ch, norm_ch, lower_ch, upper_ch, sd_ch,
        well_ch, tp_ch, well_toggle_ch, tp_toggle_ch, use_aligned_tiles_ch)
        seg_ch.view { it }
        seg_result = SEGMENTATION.out
    }
    else {
        seg_result = Channel.of(true)
    }

    if (params.DO_SEGMENTATION_MONTAGE) {
        register_flag = updatepaths_result.mix(register_result).collect()
        seg_ch = SEGMENTATION_MONTAGE(register_flag, experiment_ch, morphology_ch, seg_ch, norm_ch, lower_ch, upper_ch, sd_ch,
        well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
        seg_ch.view { it }
        seg_result = SEGMENTATION_MONTAGE.out
    }
    else {
        seg_result = Channel.of(true)
    }

    if (params.DO_CELLPOSE_SEGMENTATION && !params.DO_SEGMENTATION) {
        register_flag = updatepaths_result.mix(register_result).collect()

        cellpose_ch = CELLPOSE(register_flag, experiment_ch, batch_size_cellpose_ch, cell_diameter_ch, flow_threshold_ch, 
        cell_probability_ch, model_type_ch, morphology_ch, seg_ch, lower_ch, upper_ch, sd_ch,
        well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
        cellpose_ch.view { it }
        cellpose_result = CELLPOSE.out
    }
    else {
        cellpose_result = Channel.of(true)
    }

    if (params.DO_PUNCTA_SEGMENTATION){
        seg_flag = seg_result.mix(cellpose_result).collect()
        PUNCTA(seg_flag, experiment_ch, puncta_seg_ch, puncta_manual_thresh_ch,sigma1_ch, sigma2_ch, 
        morphology_ch, puncta_target_channel_ch,
         well_ch, tp_ch, well_toggle_ch, tp_toggle_ch, tile_ch)
        puncta_result = PUNCTA.out
    }
    else{
        puncta_result = Channel.of(true)
    }

    if (params.DO_TRACKING) {
        seg_flag = seg_result.mix(cellpose_result).collect()
        track_ch = TRACKING(seg_flag, experiment_ch, distance_threshold_ch, voronoi_bool_ch, well_ch, tp_ch, morphology_ch,
        well_toggle_ch, tp_toggle_ch)
        track_result = TRACKING.out    }
    else {
        track_result = Channel.of(true)
    }  

    if (params.DO_MONTAGE) {
        montage_flag = seg_result.mix(cellpose_result).mix(track_result).collect()
        montage_ch = MONTAGE(montage_flag, experiment_ch, tiletype_ch, montage_pattern_ch, well_ch, tp_ch, channel_ch,
        well_toggle_ch, tp_toggle_ch, channel_toggle_ch,image_overlap_ch)
        montage_result = MONTAGE.out
    }

    if (params.DO_ALIGNMENT) {
        alignment_ch = ALIGNMENT(experiment_ch,  well_ch, tp_ch, morphology_ch, alignment_algoritm_ch, robo_num_ch, dir_structure_ch, imaging_mode_ch, aligntiletype_ch, shift_dict_ch)
        alignment_result = ALIGNMENT.out
    }
    else {
        alignment_result = Channel.of(true)
    }

    if (params.DO_ALIGN_TILES_DFT) {
        align_tiles_dft_ch = ALIGN_TILES_DFT(
            experiment_ch,
            morphology_ch,
            well_ch,
            tp_ch,
            channel_ch,
            well_toggle_ch,
            tp_toggle_ch,
            channel_toggle_ch,
            tile_ch,
            shift_dict_ch
        )
        align_tiles_dft_ch.view { it }
        align_tiles_dft_result = ALIGN_TILES_DFT.out
} else {
    align_tiles_dft_result = Channel.of(true)
}

if (params.DO_ALIGN_MONTAGE_DFT) {
    align_montage_ch = Channel.of(true)
        align_montage_dft_ch = ALIGN_MONTAGE_DFT(
            align_montage_ch,
            experiment_ch,
            morphology_ch,
            well_ch,
            tp_ch,
            channel_ch,
            well_toggle_ch,
            tp_toggle_ch,
            channel_toggle_ch,
            tile_ch,
            shift_dict_ch
        )
        align_montage_dft_ch.view { it }
        align_montage_dft_result = ALIGN_MONTAGE_DFT.out
} else {
    align_montage_dft_result = Channel.of(true)
}
if (params.DO_PLATEMONTAGE) {
    montage_flag = seg_result.mix(cellpose_result).mix(track_result).collect()
    platemontage_ch = PLATEMONTAGE(montage_flag, experiment_ch, well_size_for_platemontage_ch, norm_intensity_ch, tiletype_ch, montage_pattern_ch, well_ch, tp_ch, chosen_channels_for_register_exp_ch,
    well_toggle_ch, tp_toggle_ch, channel_toggle_ch)
    montage_result = PLATEMONTAGE.out
}
else {
    montage_result = Channel.of(true)
}

if (params.DO_TRACKING_MONTAGE) {
    // prepare your input‑ready flag
    tracking_montage_ch = Channel.of(true)
    //tracking_montage_flag = seg_result.mix(cellpose_result).collect()

    // convert list to comma-separated string if needed
        // target_channel_ch is already defined globally

    combined_ch = tracking_montage_ch
    .combine(experiment_ch)
    .combine(track_type_ch)
    .combine(distance_threshold_ch)
    .combine(well_ch)
    .combine(target_channel_ch)
    .combine(motion_ch)

    // invoke the process; this returns a channel of all stdout lines
    track_montage_ch = TRACKING_MONTAGE(combined_ch)

    // print each line your Python script outputs
    track_montage_ch.view { line ->
        println line
    }

    // wire it downstream
    tracking_montage_result = track_montage_ch
}
else {
    tracking_montage_result = Channel.of(true)
}
    

if (params.DO_INTENSITY) {
    intensity_flag = seg_result.mix(cellpose_result).mix(track_result).collect()
    int_ch = INTENSITY(intensity_flag, experiment_ch, norm_ch, morphology_ch, target_channel_ch, well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
    intensity_result = INTENSITY.out
    int_ch.view { it }
}
else {
    intensity_result = Channel.of(true)
}
   
if (params.DO_CROP) {
    crop_ch = CROP(track_result, experiment_ch, target_channel_crop_ch, morphology_ch, crop_size_ch, well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
    crop_ch.view { it }
    crop_result = CROP.out
}
else {
    crop_result = Channel.of(true)
}
if (params.DO_MASK_CROP) {
    crop_mask_ch = CROP_MASK(track_result, experiment_ch, target_channel_crop_ch, morphology_ch, crop_size_ch, 
                            well_ch, tp_ch, well_toggle_ch, tp_toggle_ch)
    crop_mask_ch.view { it }
    crop_mask_result = CROP_MASK.out
} else {
    crop_mask_result = Channel.of(true)
}    
if (params.DO_CNN) {
    cnn_ch = CNN(crop_result, experiment_ch, cnn_model_type_ch, label_type_ch, label_name_ch, classes_ch, img_norn_name_cnn_ch, filters_ch, num_channels_ch, n_samples_ch,
    epochs_ch, batch_size_ch, learning_rate_ch, momentum_ch, optimizer_ch, well_ch, tp_ch, chosen_channels_for_cnn_ch,
    well_toggle_ch, tp_toggle_ch, channel_toggle_ch)
    cnn_result = CNN.out
}
else {
    cnn_result = Channel.of(true)
}

if (params.DO_GET_CSVS) {
    csv_ready = updatepaths_result.mix(register_result).mix(seg_result).mix(cellpose_result).mix(puncta_result).mix(track_result).mix(intensity_result).mix(crop_result).mix(cnn_result).collect()
    csv_ch = GETCSVS(csv_ready, experiment_ch)
    csv_ch.view { it }
}
// Run OVERLAY only if enabled--KS edit for overlay
if (params.DO_OVERLAY) {
    overlay_ch = OVERLAY(
        track_result,
        experiment_ch,
        morphology_ch,
        well_ch,
        tp_ch,
        well_toggle_ch,
        tp_toggle_ch,
        channel_toggle_ch,
        params.shift,
        params.contrast,
        tile_ch
    )
    overlay_ch.view { it } // Optional: Debugging output
}

if (params.DO_OVERLAY_MONTAGE) {
    overlay_montage_ch = Channel.of(true)
    
    overlay_ch = OVERLAY_MONTAGE(
        overlay_montage_ch,
        experiment_ch,
        morphology_ch,
        well_ch,
        tp_ch,
        well_toggle_ch,
        tp_toggle_ch,
        channel_toggle_ch,
        params.shift,
        params.contrast,
    
    )
    
}

// ***************** 2 MAIN COMMONLY USED WORKFLOWS ************************


// Read CSV platemap file -TODO Read from DB
def csv_lines = file(params.platemap_path).readLines()
def header = csv_lines[0].split(',').toList()
header = header.collect { it.trim() }
def well_index = header.indexOf('well')

if (well_index == -1) {
    error "❌ CSV platemap is missing a 'well' column."
}

// Extract all well names from the CSV (skip header)
def all_wells = csv_lines[1..-1].collect { line ->
    def cols = line.split(',')
    cols[well_index].trim()
}

// Choose wells based on params.chosen_wells
def wells_to_use = []
if (params.chosen_wells == 'all') {
    wells_to_use = all_wells
} else {
    wells_to_use = params.chosen_wells.split(',').collect { it.trim() }

    // Optional: Validate chosen wells
    wells_to_use.each { well ->
        if (!(well in all_wells)) {
            error "❌ Specified well '${well}' not found in plate map!"
        }
    }
}



// Create the well channel - use this for all workflows
well_ch = Channel.from(wells_to_use)


if (params.DO_STD_WORKFLOW) {

    log.info "\n ▶ Running OPTIMIZED DO_STD_WORKFLOW: BUNDLED_STD_WORKFLOW (MONTAGE → ALIGN_MONTAGE_DFT → SEGMENTATION → TRACKING → OVERLAY)"
   
    // Record start time for bundled workflow
    def bundle_start_time = System.currentTimeMillis()
    def bundle_start_timestamp = new Date().format("yyyy-MM-dd HH:mm:ss")
    
    // Log start time for bundled workflow
    println "🚀 [${bundle_start_timestamp}] STARTING... BUNDLED_STD_WORKFLOW for all wells: ${wells_to_use.join(', ')}"
  
    combined_bundled_std_ch = well_ch
        .combine(experiment_ch)
        .combine(tiletype_ch)
        .combine(montage_pattern_ch)
        .combine(tp_ch)
        .combine(channel_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .combine(channel_toggle_ch)
        .combine(image_overlap_ch)
        .combine(morphology_ch)
        .combine(seg_ch)
        .combine(norm_ch)
        .combine(lower_ch)
        .combine(upper_ch)
        .combine(sd_ch)
        .combine(track_type_ch)
        .combine(distance_threshold_ch)
        .combine(target_channel_ch)
        .combine(shift_ch)
        .combine(contrast_ch)
        .combine(tile_ch)
        .combine(shift_dict_ch)
        .combine(motion_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            
            // Map to the BUNDLED_STD_WORKFLOW input order:
            // exp, tiletype, montage_pattern, chosen_timepoints, chosen_channels, wells_toggle, 
            // timepoints_toggle, channels_toggle, image_overlap, morphology_channel, segmentation_method,
            // img_norm_name, lower_area_thresh, upper_area_thresh, sd_scale_factor, track_type,
            // distance_threshold, target_channel, well, shift, contrast, tile, shift_dict, motion
            
            def exp               = flat[1]
            def tiletype          = flat[2]
            def montage_pattern   = flat[3]
            def chosen_timepoints = flat[4]
            def chosen_channels   = flat[5]
            def wells_toggle      = flat[6]
            def timepoints_toggle = flat[7]
            def channels_toggle   = flat[8]
            def image_overlap     = flat[9]
            def morphology_channel = flat[10]
            def segmentation_method = flat[11]
            def img_norm_name     = flat[12]
            def lower_area_thresh = flat[13]
            def upper_area_thresh = flat[14]
            def sd_scale_factor   = flat[15]
            def track_type        = flat[16]
            def distance_threshold = flat[17]
            def target_channel    = flat[18]
            def well              = flat[0]
            def shift             = flat[19]
            def contrast          = flat[20]
            def tile              = flat[21]
            def shift_dict        = flat[22]
            def motion            = flat[23]

            return tuple(exp, tiletype, montage_pattern, chosen_timepoints, chosen_channels, wells_toggle,
                       timepoints_toggle, channels_toggle, image_overlap, morphology_channel, segmentation_method,
                       img_norm_name, lower_area_thresh, upper_area_thresh, sd_scale_factor, track_type,
                       distance_threshold, target_channel, well, shift, contrast, tile, shift_dict, motion)
        }
    
  

    bundled_std_result_ch = BUNDLED_STD_WORKFLOW(combined_bundled_std_ch)
    
    
    bundled_std_result_ch.view { tuple ->
        def (well, flag) = tuple
        def bundle_end_time = System.currentTimeMillis()
        def bundle_end_timestamp = new Date().format("yyyy-MM-dd HH:mm:ss")
        def total_time_ms = bundle_end_time - bundle_start_time
        def total_time_seconds = total_time_ms / 1000.0
        def total_time_minutes = total_time_seconds / 60.0
        
        println "🎉 [${bundle_end_timestamp}] COMPLETED... BUNDLED_STD_WORKFLOW for well: $well (MONTAGE → ALIGN_MONTAGE_DFT → SEGMENTATION → TRACKING → OVERLAY)"
        println "⏱️  Total time: ${total_time_seconds.round(1)}s (${total_time_minutes.round(2)} min)"
    }
}


if (params.DO_STD_WORKFLOW_IXM) {

    log.info "\n ▶ Running OPTIMIZED DO_STD_WORKFLOW_IXM: BUNDLED_WORKFLOW_IXM (MONTAGE → SEGMENTATION → TRACKING → OVERLAY)"
   
    // Record start time for bundled workflow
    def bundle_start_time = System.currentTimeMillis()
    def bundle_start_timestamp = new Date().format("yyyy-MM-dd HH:mm:ss")
    
    // Log start time for bundled workflow
    println "🚀 [${bundle_start_timestamp}] STARTING... BUNDLED_WORKFLOW_IXM for all wells: ${wells_to_use.join(', ')}"
  
    combined_bundled_ch = well_ch
        .combine(experiment_ch)
        .combine(tiletype_ch)
        .combine(montage_pattern_ch)
        .combine(tp_ch)
        .combine(channel_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .combine(channel_toggle_ch)
        .combine(image_overlap_ch)
        .combine(morphology_ch)
        .combine(seg_ch)
        .combine(norm_ch)
        .combine(lower_ch)
        .combine(upper_ch)
        .combine(sd_ch)
        .combine(track_type_ch)
        .combine(distance_threshold_ch)
        .combine(target_channel_ch)
        .combine(shift_ch)
        .combine(contrast_ch)
        .combine(motion_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            
            // Map to the BUNDLED_WORKFLOW_IXM input order:
            // exp, tiletype, montage_pattern, chosen_timepoints, chosen_channels, wells_toggle, 
            // timepoints_toggle, channels_toggle, image_overlap, morphology_channel, segmentation_method,
            // img_norm_name, lower_area_thresh, upper_area_thresh, sd_scale_factor, track_type,
            // distance_threshold, target_channel, well, shift, contrast, motion
            
            def exp               = flat[1]
            def tiletype          = flat[2]
            def montage_pattern   = flat[3]
            def chosen_timepoints = flat[4]
            def chosen_channels   = flat[5]
            def wells_toggle      = flat[6]
            def timepoints_toggle = flat[7]
            def channels_toggle   = flat[8]
            def image_overlap     = flat[9]
            def morphology_channel = flat[10]
            def segmentation_method = flat[11]
            def img_norm_name     = flat[12]
            def lower_area_thresh = flat[13]
            def upper_area_thresh = flat[14]
            def sd_scale_factor   = flat[15]
            def track_type        = flat[16]
            def distance_threshold = flat[17]
            def target_channel    = flat[18]
            def well              = flat[0]
            def shift             = flat[19]
            def contrast          = flat[20]
            def motion            = flat[21]

            return tuple(exp, tiletype, montage_pattern, chosen_timepoints, chosen_channels, wells_toggle,
                       timepoints_toggle, channels_toggle, image_overlap, morphology_channel, segmentation_method,
                       img_norm_name, lower_area_thresh, upper_area_thresh, sd_scale_factor, track_type,
                       distance_threshold, target_channel, well, shift, contrast, motion)
        }
    
  

    bundled_result_ch = BUNDLED_WORKFLOW_IXM(combined_bundled_ch)
    
    
    bundled_result_ch.view { tuple ->
        def (well, flag) = tuple
        def bundle_end_time = System.currentTimeMillis()
        def bundle_end_timestamp = new Date().format("yyyy-MM-dd HH:mm:ss")
        def total_time_ms = bundle_end_time - bundle_start_time
        def total_time_seconds = total_time_ms / 1000.0
        def total_time_minutes = total_time_seconds / 60.0
        
        println "🎉 [${bundle_end_timestamp}] COMPLETED... BUNDLED_WORKFLOW_IXM for well: $well (MONTAGE → SEGMENTATION → TRACKING → OVERLAY)"
        println "⏱️  Total time: ${total_time_seconds.round(1)}s (${total_time_minutes.round(2)} min)"
    }
}

if (params.DO_BUNDLED_STD_WORKFLOW) {

    log.info "\n ▶ Running OPTIMIZED DO_BUNDLED_STD_WORKFLOW: BUNDLED_STD_WORKFLOW (MONTAGE → ALIGN_MONTAGE_DFT → SEGMENTATION → TRACKING → OVERLAY)"
   
    // Record start time for bundled workflow
    def bundle_start_time = System.currentTimeMillis()
    def bundle_start_timestamp = new Date().format("yyyy-MM-dd HH:mm:ss")
    
    // Log start time for bundled workflow
    println "🚀 [${bundle_start_timestamp}] STARTING... BUNDLED_STD_WORKFLOW for all wells: ${wells_to_use.join(', ')}"
  
    combined_bundled_std_ch = well_ch
        .combine(experiment_ch)
        .combine(tiletype_ch)
        .combine(montage_pattern_ch)
        .combine(tp_ch)
        .combine(channel_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .combine(channel_toggle_ch)
        .combine(image_overlap_ch)
        .combine(morphology_ch)
        .combine(seg_ch)
        .combine(norm_ch)
        .combine(lower_ch)
        .combine(upper_ch)
        .combine(sd_ch)
        .combine(track_type_ch)
        .combine(distance_threshold_ch)
        .combine(target_channel_ch)
        .combine(shift_ch)
        .combine(contrast_ch)
        .combine(tile_ch)
        .combine(shift_dict_ch)
        .combine(motion_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            
            // Map to the BUNDLED_STD_WORKFLOW input order:
            // exp, tiletype, montage_pattern, chosen_timepoints, chosen_channels, wells_toggle, 
            // timepoints_toggle, channels_toggle, image_overlap, morphology_channel, segmentation_method,
            // img_norm_name, lower_area_thresh, upper_area_thresh, sd_scale_factor, track_type,
            // distance_threshold, target_channel, well, shift, contrast, tile, shift_dict, motion
            
            def exp               = flat[1]
            def tiletype          = flat[2]
            def montage_pattern   = flat[3]
            def chosen_timepoints = flat[4]
            def chosen_channels   = flat[5]
            def wells_toggle      = flat[6]
            def timepoints_toggle = flat[7]
            def channels_toggle   = flat[8]
            def image_overlap     = flat[9]
            def morphology_channel = flat[10]
            def segmentation_method = flat[11]
            def img_norm_name     = flat[12]
            def lower_area_thresh = flat[13]
            def upper_area_thresh = flat[14]
            def sd_scale_factor   = flat[15]
            def track_type        = flat[16]
            def distance_threshold = flat[17]
            def target_channel    = flat[18]
            def well              = flat[0]
            def shift             = flat[19]
            def contrast          = flat[20]
            def tile              = flat[21]
            def shift_dict        = flat[22]
            def motion            = flat[23]

            return tuple(exp, tiletype, montage_pattern, chosen_timepoints, chosen_channels, wells_toggle,
                       timepoints_toggle, channels_toggle, image_overlap, morphology_channel, segmentation_method,
                       img_norm_name, lower_area_thresh, upper_area_thresh, sd_scale_factor, track_type,
                       distance_threshold, target_channel, well, shift, contrast, tile, shift_dict, motion)
        }
    
  

    bundled_std_result_ch = BUNDLED_STD_WORKFLOW(combined_bundled_std_ch)
    
    
    bundled_std_result_ch.view { tuple ->
        def (well, flag) = tuple
        def bundle_end_time = System.currentTimeMillis()
        def bundle_end_timestamp = new Date().format("yyyy-MM-dd HH:mm:ss")
        def total_time_ms = bundle_end_time - bundle_start_time
        def total_time_seconds = total_time_ms / 1000.0
        def total_time_minutes = total_time_seconds / 60.0
        
        println "🎉 [${bundle_end_timestamp}] COMPLETED... BUNDLED_STD_WORKFLOW for well: $well (MONTAGE → ALIGN_MONTAGE_DFT → SEGMENTATION → TRACKING → OVERLAY)"
        println "⏱️  Total time: ${total_time_seconds.round(1)}s (${total_time_minutes.round(2)} min)"
    }
}


}

workflow.onComplete {
    log.info(workflow.success ? "\n The pipeline is successfully completed! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : 'Oops .. something went wrong')
}

