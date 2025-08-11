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
well_ch = Channel.of(params.chosen_wells)
tile_ch = Channel.of(params.tile)
use_aligned_tiles_ch = Channel.of(params.use_aligned_tiles)
tp_ch = Channel.of(params.chosen_timepoints)
channel_ch = Channel.of(params.chosen_channels)
well_toggle_ch = Channel.of(params.wells_toggle)
tp_toggle_ch = Channel.of(params.timepoints_toggle)
channel_toggle_ch = Channel.of(params.channels_toggle)
image_overlap_ch = Channel.of(params.image_overlap) //austin added "_ch" 5/17
target_channel_ch = Channel.from(params.target_channel)
target_channel_crop_ch = Channel.from(params.target_channel_crop)
puncta_target_channel_ch = Channel.from(params.puncta_target_channel)
morphology_ch = Channel.of(params.morphology_channel)
distance_threshold_ch = Channel.of(params.distance_threshold)
voronoi_bool_ch = Channel.of(params.voronoi_bool)
track_type_ch           = Channel.of(params.track_type)
crop_size_ch = Channel.of(params.crop_size)
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
    CROP; CROP_MASK; MONTAGE; PLATEMONTAGE; CNN; GETCSVS; BASHEX; UPDATEPATHS; NORMALIZATION; COPY_MASK_TO_TRACKED; OVERLAY_MONTAGE} from './modules.nf'

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
    def target_channel_str = params.target_channel instanceof List 
        ? params.target_channel.join(',') 
        : params.target_channel

    // create a value channel from the string
    target_channel_ch = Channel.value(target_channel_str)

    combined_ch = tracking_montage_ch
    .combine(experiment_ch)
    .combine(track_type_ch)
    .combine(distance_threshold_ch)
    .combine(well_ch)
    .combine(target_channel_ch)

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
println "First line of CSV: ${csv_lines[0]}"
def header = csv_lines[0].split(',').toList()
println "Header columns (raw): ${header} - type: ${header.getClass()}"
header = header.collect { it.trim() }
println "Header columns (trimmed): ${header} - type: ${header.getClass()}"

def well_index = header.indexOf('well')
println "Index of 'well': ${well_index}"

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



// Create the well channel
well_ch = Channel.from(wells_to_use)
well_ch.view { "💧 well_ch emits: $it" }

println "✅ Wells to use: ${wells_to_use}"



if (params.DO_STD_WORKFLOW) {

    log.info "▶ Running DO_STD_WORKFLOW: MONTAGE → ALIGN_MONTAGE_DFT → SEGMENTATION_MONTAGE → TRACKING_MONTAGE → OVERLAY_MONTAGE"

    // Step 1: MONTAGE
    montage_ready_ch = Channel.of(true)

    combined_montage_ch = well_ch
        .combine(montage_ready_ch)
        .combine(experiment_ch)
        .combine(tiletype_ch)
        .combine(montage_pattern_ch)
        .combine(tp_ch)
        .combine(channel_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .combine(channel_toggle_ch)
        .combine(image_overlap_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            def ready       = flat[1]
            def exp         = flat[2]
            def tiletype    = flat[3]
            def montage_pat = flat[4]
            def well        = flat[0]
            def tp          = flat[5]
            def ch          = flat[6]
            def well_toggle = flat[7]
            def tp_toggle   = flat[8]
            def ch_toggle   = flat[9]
            def img_overlap = flat[10]

            return tuple(ready, exp, tiletype, montage_pat, well, tp, ch, well_toggle, tp_toggle, ch_toggle, img_overlap)
        }

    montage_result_ch = MONTAGE(combined_montage_ch)

    montage_result_ch.view { tuple ->
        def (flag, well) = tuple
        println "🧪 Emitted from MONTAGE: $flag (flag: $well)"
    }


    // Step 2: ALIGN_MONTAGE_DFT
    align_ready_ch = montage_result_ch

    combined_align_ch = align_ready_ch
        .combine(experiment_ch)
        .combine(morphology_ch)
        .combine(well_ch)
        .combine(tp_ch)
        .combine(channel_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .combine(channel_toggle_ch)
        .combine(tile_ch)
        .combine(shift_dict_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            def flag        = flat[0]
            def exp         = flat[1]
            def morph       = flat[2]
            def well        = flat[3]
            def tp          = flat[4]
            def ch          = flat[5]
            def well_toggle = flat[6]
            def tp_toggle   = flat[7]
            def ch_toggle   = flat[8]
            def tile        = flat[9]
            def shift_dict  = flat[10]

            return tuple(flag, exp, morph, well, tp, ch, well_toggle, tp_toggle, ch_toggle, tile, shift_dict)
        }

    align_result_ch = ALIGN_MONTAGE_DFT(combined_align_ch)
    align_result_ch.view { tuple ->
        def (flag, well) = tuple
        println "🧪 Emitted from ALIGN_MONTAGE_DFT: $flag (flag: $well)"
    }


    // Step 3: SEGMENTATION_MONTAGE
    segmont_ready_ch = align_result_ch

    combined_segmont_ch = segmont_ready_ch
        .combine(experiment_ch)
        .combine(morphology_ch)
        .combine(seg_ch)
        .combine(norm_ch)
        .combine(lower_ch)
        .combine(upper_ch)
        .combine(sd_ch)
        .combine(well_ch)
        .combine(tp_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            def flag        = flat[0]
            def exp         = flat[1]
            def morph       = flat[2]
            def seg_method  = flat[3]
            def norm        = flat[4]
            def lower       = flat[5]
            def upper       = flat[6]
            def sd          = flat[7]
            def well        = flat[8]
            def tp          = flat[9]
            def well_toggle = flat[10]
            def tp_toggle   = flat[11]

            return tuple(flag, exp, morph, seg_method, norm, lower, upper, sd, well, tp, well_toggle, tp_toggle)
        }

    segmont_result_ch = SEGMENTATION_MONTAGE(combined_segmont_ch)
    segmont_result_ch.view { tuple ->
        def (flag, well) = tuple
        println "🧪 Emitted from SEGMENTATION: $flag (flag: $well)"
    }


    // Step 4: TRACKING_MONTAGE
    trackmont_ready_ch = segmont_result_ch

    def target_channel_str = params.target_channel instanceof List 
        ? params.target_channel.join(',')
        : params.target_channel
    target_channel_ch = Channel.value(target_channel_str)

    combined_trackmont_ch = trackmont_ready_ch
        .combine(experiment_ch)
        .combine(track_type_ch)
        .combine(distance_threshold_ch)
        .combine(well_ch)
        .combine(target_channel_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            def flag        = flat[0]
            def exp         = flat[1]
            def track_type  = flat[2]
            def dist_thresh = flat[3]
            def well        = flat[4]
            def target_ch   = flat[5]

            return tuple(flag, exp, track_type, dist_thresh, well, target_ch)
        }

    trackmont_result_ch = TRACKING_MONTAGE(combined_trackmont_ch)
    trackmont_result_ch.view { tuple ->
        def (flag, well) = tuple
        println "🧪 Emitted from TRACKING: $flag (flag: $well)"
    }


    // Step 5: OVERLAY_MONTAGE
    overlay_ready_ch = trackmont_result_ch

    combined_overlay_ch = overlay_ready_ch
        .combine(experiment_ch)
        .combine(morphology_ch)
        .combine(well_ch)
        .combine(tp_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .combine(channel_toggle_ch)
        .combine(shift_ch)
        .combine(contrast_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            def flag        = flat[0]
            def exp         = flat[1]
            def morph       = flat[2]
            def well        = flat[3]
            def tp          = flat[4]
            def well_toggle = flat[5]
            def tp_toggle   = flat[6]
            def ch_toggle   = flat[7]
            def shift       = flat[8]
            def contrast    = flat[9]

            return tuple(flag, exp, morph, well, tp, well_toggle, tp_toggle, ch_toggle, shift, contrast)
        }

    overlay_result_ch = OVERLAY_MONTAGE(combined_overlay_ch)
    overlay_result_ch.view { tuple ->
        def (flag, well) = tuple
        println "🧪 Emitted from OVERLAY: $flag (flag: $well)"
    }
}


if (params.DO_STD_WORKFLOW_IXM) {

    log.info "▶ Running DO_STD_WORKFLOW_IXM: MONTAGE → SEGMENTATION_MONTAGE → TRACKING_MONTAGE → OVERLAY_MONTAGE"

    // Step 1: MONTAGE
    // Combine each well with all singleton params
    montage_ready_ch = Channel.of(true)

    combined_montage_ch = well_ch
        .combine(montage_ready_ch)
        .combine(experiment_ch)
        .combine(tiletype_ch)
        .combine(montage_pattern_ch)
        .combine(tp_ch)
        .combine(channel_ch)
        .combine(well_toggle_ch)
        .combine(tp_toggle_ch)
        .combine(channel_toggle_ch)
        .combine(image_overlap_ch)
        .map { nestedTuple ->
            def flat = nestedTuple.flatten()
            // Reorder flat list to process input order:
            // Current process input order:
            // ready, exp, tiletype, montage_pattern, well, chosen_timepoints, chosen_channels, wells_toggle, timepoints_toggle, channels_toggle, image_overlap
            
            def ready       = flat[1]
            def exp         = flat[2]
            def tiletype    = flat[3]
            def montage_pattern = flat[4]
            def well        = flat[0]
            def chosen_timepoints = flat[5]
            def chosen_channels   = flat[6]
            def wells_toggle      = flat[7]
            def timepoints_toggle = flat[8]
            def channels_toggle   = flat[9]
            def image_overlap     = flat[10]

            return tuple(ready, exp, tiletype, montage_pattern, well, chosen_timepoints, chosen_channels, wells_toggle, timepoints_toggle, channels_toggle, image_overlap)
        }

    montage_result_ch = MONTAGE(combined_montage_ch)
    montage_result_ch.view { tuple ->
    def (flag, well) = tuple
    println "🧪 Emitted from MONTAGE: $flag (flag: $well)"
    }
 

    // Step 2: SEGMENTATION_MONTAGE
    segmont_ready_ch = montage_result_ch

    combined_segmont_ch = segmont_ready_ch
    .combine(experiment_ch)
    .combine(morphology_ch)
    .combine(seg_ch)
    .combine(norm_ch)
    .combine(lower_ch)
    .combine(upper_ch)
    .combine(sd_ch)
    .combine(tp_ch)
    .combine(well_toggle_ch)
    .combine(tp_toggle_ch)
    .map { nestedTuple ->
        def flat = nestedTuple.flatten()
        // flat should have 12 elements corresponding to the segmentation inputs:
        // ready, well, exp, morphology_channel, segmentation_method, img_norm_name,
        // lower_area_thresh, upper_area_thresh, sd_scale_factor, chosen_timepoints,
        // wells_toggle, timepoints_toggle

        def ready         = flat[0]  // from montage_ch: true
        def well          = flat[1]  // from montage_ch: D03 or D05
        def exp           = flat[2]
        def morphology    = flat[3]
        def seg_params    = flat[4]
        def norm          = flat[5]
        def lower         = flat[6]
        def upper         = flat[7]
        def sd            = flat[8]
        def chosen_tp     = flat[9]
        def wells_toggle  = flat[10]
        def tp_toggle     = flat[11]

        return tuple(ready, exp, morphology, seg_params, norm, lower, upper, sd, well, chosen_tp, wells_toggle, tp_toggle)
    }

    segmentation_montage_result_ch = SEGMENTATION_MONTAGE(combined_segmont_ch)

    segmentation_montage_result_ch.view { tuple ->
    def (flag, well) = tuple
    println "🧪 Emitted from SEGMENTATION: $flag (flag: $well)"
    }
   

    // Step 4: TRACKING_MONTAGE

    tracking_ready_ch = segmentation_montage_result_ch

    def target_channel_str = params.target_channel instanceof List 
        ? params.target_channel.join(',')
        : params.target_channel
    target_channel_ch = Channel.value(target_channel_str)


    combined_trackmont_ch = tracking_ready_ch
    .combine(experiment_ch)
    .combine(track_type_ch)
    .combine(distance_threshold_ch)
    .combine(target_channel_ch)
    .map { nestedTuple ->
        def flat = nestedTuple.flatten()
        // flat should contain: ready, well, experiment, track_type, distance_threshold, target_channel
        def ready         = flat[0]
        def exp           = flat[2]
        def track_type    = flat[3]
        def dist_thresh   = flat[4]
        def well          = flat[1]
        def target_ch     = flat[5]

        return tuple(ready, exp, track_type, dist_thresh, well, target_ch)
    }

    tracking_montage_result_ch = TRACKING_MONTAGE(combined_trackmont_ch)
    tracking_montage_result_ch.view { tuple ->
    def (flag, well) = tuple
    println "🧪 Emitted from TRACKING: $flag (flag: $well)"
    }


    // Step 5: OVERLAY_MONTAGE
    overlay_ready_ch = tracking_montage_result_ch

    combined_overlay_ch = overlay_ready_ch
    .combine(experiment_ch)
    .combine(morphology_ch)
    .combine(tp_ch)
    .combine(well_toggle_ch)
    .combine(tp_toggle_ch)
    .combine(channel_toggle_ch)
    .combine(shift_ch)
    .combine(contrast_ch)
    .map { nestedTuple ->
        def flat = nestedTuple.flatten()
        // flat order corresponds to inputs of OVERLAY_MONTAGE process input tuple
        def flag           = flat[0]
        def exp            = flat[2]
        def morphology     = flat[3]
        def well           = flat[1]
        def tp             = flat[4]
        def well_toggle    = flat[5]
        def tp_toggle      = flat[6]
        def channel_toggle = flat[7]
        def shift          = flat[8]
        def contrast       = flat[9]

        return tuple(flag, exp, morphology, well, tp, well_toggle, tp_toggle, channel_toggle, shift, contrast)
    }

    overlay_result_ch = OVERLAY_MONTAGE(combined_overlay_ch)
    overlay_result_ch.view { tuple ->
    def (flag, well) = tuple
    println "🧪 Emitted from OVERLAY: $flag (flag: $well)"
    }
}


}

workflow.onComplete {
    log.info(workflow.success ? "\n The pipeline is successfully completed! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : 'Oops .. something went wrong')
}

