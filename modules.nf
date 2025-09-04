#!/usr/bin/env nextflow

params.greeting = 'Hello world!'
greeting_ch = Channel.of(params.greeting)



process REGISTER_EXPERIMENT {
    // containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"

    input:
    val input_path
    val output_path
    val template_path
    val platemap_path
    val ixm_hts_file
    val robo_file
    val overwrite_experiment       
    val robo_num                   
    val illumination_file         

    val chosen_wells
    val chosen_timepoints
    val chosen_channels
    val wells_toggle
    val timepoints_toggle
    val channels_toggle

    output:
    val true

    script:
    """
    register_experiment.py --input_path ${input_path} --output_path ${output_path} --template_path ${template_path} \
    --platemap_path ${platemap_path} --ixm_hts_file ${ixm_hts_file} --robo_file ${robo_file} --overwrite_experiment ${overwrite_experiment} \
    --robo_num ${robo_num} \
    --illumination_file ${illumination_file} \
    --chosen_wells ${chosen_wells} --chosen_channels ${chosen_channels} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --channels_toggle ${channels_toggle} --timepoints_toggle ${timepoints_toggle}
    """
}


process SEGMENTATION {
    //containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val ready
    val exp
    val morphology_channel
    val segmentation_method
    val img_norm_name
    val lower_area_thresh
    val upper_area_thresh
    val sd_scale_factor
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle
    val use_aligned_tiles

    output: 
    val true

    script:
    """
    segmentation.py --experiment ${exp} --segmentation_method ${segmentation_method} \
    --img_norm_name ${img_norm_name}  --lower_area_thresh ${lower_area_thresh} --upper_area_thresh ${upper_area_thresh} \
    --sd_scale_factor ${sd_scale_factor} \
    --chosen_wells ${chosen_wells} --chosen_channels ${morphology_channel} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} \
   ${ use_aligned_tiles ? '--use_aligned_tiles' : '' }
    """
}



process NORMALIZATION {

    input:
    val ready
    val exp
    val img_norm_name
    val chosen_wells
    val chosen_channels
    val chosen_timepoints
    val wells_toggle
    val channels_toggle
    val timepoints_toggle

    output:
    val true


    script:
    """
    echo "Activating Conda environment inside Singularity container..."
    source /opt/conda/bin/activate datastudy
    echo "Activated Conda environment inside Singularity container..."
    pip install numpy
    normalization.py --experiment ${exp} --img_norm_name ${img_norm_name} --chosen_wells ${chosen_wells} --chosen_channels ${chosen_channels} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --channels_toggle ${channels_toggle} --timepoints_toggle ${timepoints_toggle} 
    """
}

process CELLPOSE {
    containerOptions "--gpus all --mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val ready
    val exp
    val batch_size
    val cell_diameter
    val flow_threshold
    val cell_probability
    val model_type
    val morphology_channel
    val segmentation_method
    val lower_area_thresh
    val upper_area_thresh
    val sd_scale_factor
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle

    output:
    val true

    script:
    """
    cellpose_segmentation.py --experiment ${exp} --batch_size ${batch_size} --cell_diameter ${cell_diameter} --flow_threshold ${flow_threshold} \
    --cell_probability ${cell_probability} --model_type ${model_type} \
    --chosen_channels ${morphology_channel} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
    """
}

process PUNCTA {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val ready
    val exp
    val segmentation_method
    val manual_thresh
    val sigma1
    val sigma2
    val morphology_channel
    each target_channel
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle
    val tile

    output:
    val true

    script:
    """
    puncta.py --experiment ${exp} --segmentation_method ${segmentation_method} --manual_thresh ${manual_thresh} \
    --sigma1 ${sigma1} --sigma2 ${sigma2} \
    --chosen_channels ${morphology_channel} --target_channel ${target_channel} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} \
    --tile ${tile}
    """
}

process TRACKING {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/ --mount type=bind,src=/opt/gurobi/,target=/opt/gurobi/"
    input:
    val ready
    val exp
    val distance_threshold
    val voronoi_bool
    val chosen_wells
    val chosen_timepoints
    val morphology_channel
    val wells_toggle
    val timepoints_toggle


    output:
    val true

    script:
    """
    echo "Activating Conda environment inside Singularity container..."
    source /opt/conda/bin/activate datastudy
    echo "Activated Conda environment inside Singularity container..."
    conda install -c conda-forge pyomo

    tracking.py --experiment ${exp} --distance_threshold ${distance_threshold} --VORONOI_BOOL ${voronoi_bool} \
    --chosen_channels ${morphology_channel} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
    """
}



process INTENSITY {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    cpus 4
    input:
    val ready
    val exp
    val img_norm_name
    val morphology_channel
    each target_channel
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle

    output:
    val true

    script:
    """
    intensity.py --experiment ${exp} --img_norm_name ${img_norm_name}  \
    --morphology_channel ${morphology_channel} --target_channel ${target_channel} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
    """
}

process COPY_MASK_TO_TRACKED {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    
    input:
    val ready
    val experiment
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle
    val tile

    output:
    val true

    script:
    """
    mask_to_masktracked.py --experiment ${experiment} \
        --chosen_wells ${chosen_wells} \
        --chosen_timepoints ${chosen_timepoints} \
        --wells_toggle ${wells_toggle} \
        --timepoints_toggle ${timepoints_toggle} \
        --tile ${tile}
    """
}

//original crop
process CROP {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    memory '2 GB'
    cpus 4
    input:
    val ready
    val exp
    each target_channel_crop
    val morphology_channel
    val crop_size
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle

    output:
    val true

    script:
    """
    crop.py --experiment ${exp} --crop_size ${crop_size} \
    --target_channel ${target_channel_crop} \
    --chosen_wells ${chosen_wells} --chosen_channels ${morphology_channel} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
    """
}
process CROP_MASK {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    memory '2 GB'
    cpus 4
    input:
    val ready
    val exp
    each target_channel_crop
    val morphology_channel
    val crop_size
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle

    output:
    val true

    script:
    """
    crop_mask.py --experiment ${exp} --crop_size ${crop_size} \
    --target_channel ${target_channel_crop} \
    --chosen_wells ${chosen_wells} --chosen_channels ${morphology_channel} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
    """
}


process ALIGNMENT {
     // Force bash shell so `conda activate` works
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val exp
    val chosen_wells
    val chosen_timepoints
    val morphology_channel
    val alignment_algorithm
    val robo_num
    val dir_structure
    val imaging_mode
    val aligntiletype
    val shift
    
    output:
    val true

    script:
    """

    align_tiles_dft.py --experiment ${exp} --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --morphology_channel ${morphology_channel} --alignment_algorithm ${alignment_algorithm} --robo_num ${robo_num} \
    --dir_structure ${dir_structure} --imaging_mode ${imaging_mode} --tiletype ${aligntiletype} --shift ${shift}
    """
}


process ALIGN_TILES_DFT {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    tag "$experiment"
    publishDir "$params.outdir/AlignedTiles", mode: 'copy'

    input:
    val experiment
    val morphology_channel
    val wells
    val timepoints
    val channels
    val wells_toggle
    val timepoints_toggle
    val channels_toggle
    val tile
    val shift_dict

    output:
    path "AlignedTiles/**", optional: true

    script:
    """

    align_tiles_dft.py \
      --experiment "$experiment" \
      --morphology_channel "$morphology_channel" \
      --chosen_wells "$wells" \
      --chosen_timepoints "$timepoints" \
      --chosen_channels "$channels" \
      --wells_toggle "$wells_toggle" \
      --timepoints_toggle "$timepoints_toggle" \
      --channels_toggle "$channels_toggle" \
      --tile "$tile" \
      --shift_dict "$shift_dict" 
    """
}


process PLATEMONTAGE {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val ready
    val exp
    val img_size
    val norm_intensity
    val tiletype
    val montage_pattern
    val chosen_wells
    val chosen_timepoints
    val chosen_channels
    val wells_toggle
    val timepoints_toggle
    val channels_toggle



    output:
    stdout

    script:
    """
    plate_montage.py --experiment ${exp} --img_size ${img_size} --norm_intensity ${norm_intensity} --tiletype ${tiletype} --montage_pattern ${montage_pattern} \
    --chosen_channels ${chosen_channels} --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --channels_toggle ${channels_toggle} --timepoints_toggle ${timepoints_toggle}
    """
}

process CNN {
    containerOptions "--gpus all --mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    maxForks = 1

    input:
    val ready
    val exp
    val model_type
    val label_type
    val label_name
    val classes
    val img_norm_name
    val filters
    val num_channels
    val n_samples
    val epochs
    val batch_size
    val learning_rate
    val momentum
    val optimizer
    val chosen_wells
    val chosen_timepoints
    val chosen_channels
    val wells_toggle
    val timepoints_toggle
    val channels_toggle

    output:
    val true

    script:
    """
    cnn.py --experiment ${exp} --model_type ${model_type} --label_type ${label_type} --label_name ${label_name} --classes ${classes} \
    --img_norm_name ${img_norm_name} --filters ${filters} --num_channels ${num_channels} --n_samples ${n_samples} \
    --epochs ${epochs} --batch_size ${batch_size} --learning_rate ${learning_rate} \
    --momentum ${momentum} --optimizer ${optimizer} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} --chosen_channels ${chosen_channels}\
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} --channels_toggle ${channels_toggle}
    """
}

//Original
process GETCSVS {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val ready
    val exp

    output:
    val true

    script:
    """
    get_csvs.py --experiment ${exp}
    """
}

process UPDATEPATHS {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val exp

    output:
    val true

    script:
    """
    update_path.py --experiment ${exp}
    """
}

process BASHEX {
    tag "Bash Script Test"
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/lab/GALAXY_INFO,target=/gladstone/finkbeiner/lab/GALAXY_INFO"
    input:
    val x

    output:
    stdout
    
    script:
    """
    ls '/gladstone/finkbeiner/lab/GALAXY_INFO/'
    """
}

process SPLITLETTERS {
    input:
    val x

    output:
    path 'chunk_*'

    """
    printf '$x' | split -b 6 - chunk_
    """
}

process CONVERTTOUPPER {
    input:
    path y

    output:
    stdout

    """
    cat $y | tr '[a-z]' '[A-Z]' 
    """
}

workflow {
    letters_ch = SPLITLETTERS(greeting_ch)
    results_ch = CONVERTTOUPPER(letters_ch.flatten())
    results_ch.view{ it }
}
// KS edit to include overlays
process OVERLAY {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    
    input:
    val ready
    val exp
    val morphology_channel
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle
    val channels_toggle
    val shift
    val contrast
    val tile

    output:
    val true

    script:
    """
    overlay.py --experiment ${exp} --target_channel ${morphology_channel} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} \
    --channels_toggle ${channels_toggle} --shift ${shift} --contrast ${contrast} \
    --tile ${tile}
    """
}




// *****************

process MONTAGE {
    // containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    tag "$well"

    input:
    tuple val(ready),
      val(exp),
      val(tiletype),
      val(montage_pattern),
      val(well),
      val(chosen_timepoints),
      val(chosen_channels),
      val(wells_toggle),
      val(timepoints_toggle),
      val(channels_toggle),
      val(image_overlap)


    output:
    tuple  val(true), val(well)


    script:
    """
    set -eux
    montage.py --experiment ${exp} --tiletype ${tiletype} --montage_pattern ${montage_pattern} \
    --chosen_channels ${chosen_channels} --chosen_wells ${well} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --channels_toggle ${channels_toggle} --timepoints_toggle ${timepoints_toggle} \
    --image_overlap ${image_overlap}
    """
}

process ALIGN_MONTAGE_DFT {
    //containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    tag "$well"
    

    input:
      tuple val(ready), 
          val(experiment), 
          val(morphology_channel), 
          val(well), 
          val(timepoints), 
          val(channels), 
          val(wells_toggle), 
          val(timepoints_toggle), 
          val(channels_toggle), 
          val(tile), 
          val(shift_dict)

    output:
    tuple val(true), val(well)

    script:
    """
    align_montage_dft.py \
      --experiment "$experiment" \
      --morphology_channel "$morphology_channel" \
      --chosen_wells "$well" \
      --chosen_timepoints "$timepoints" \
      --chosen_channels "$channels" \
      --wells_toggle "$wells_toggle" \
      --timepoints_toggle "$timepoints_toggle" \
      --channels_toggle "$channels_toggle" \
      --tile "$tile" \
      --shift_dict "$shift_dict" 
    """
}


process SEGMENTATION_MONTAGE {
    //containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"

    tag "$well"
    // publishDir "$params.outdir/CellMasksMontage/${well}", mode: 'copy'

    input:
    tuple val(ready),
          val(exp),
          val(morphology_channel),
          val(segmentation_method),
          val(img_norm_name),
          val(lower_area_thresh),
          val(upper_area_thresh),
          val(sd_scale_factor),
          val(well),                    
          val(chosen_timepoints),
          val(wells_toggle),
          val(timepoints_toggle)
   

    output: 
    tuple val(true), val(well) 


    script:
    """
    segmentation_montage.py --experiment ${exp} --segmentation_method ${segmentation_method} \
    --img_norm_name ${img_norm_name}  --lower_area_thresh ${lower_area_thresh} --upper_area_thresh ${upper_area_thresh} \
    --sd_scale_factor ${sd_scale_factor} \
    --chosen_wells ${well} --chosen_channels ${morphology_channel} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}  
    """
}

process TRACKING_MONTAGE {

    tag "$well"
    echo = true
    
    //containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"

    input:
    tuple val(ready), 
          val(exp), 
          val(track_type), 
          val(distance_threshold), 
          val(well), 
          val(target_channel) 

    output:
    tuple val(true), val(well)


    script:
    """
    tracking_montage.py --experiment  ${exp} --track_type  ${track_type} --max_dist    ${distance_threshold} --wells ${well} --target_channel ${target_channel } 
    """
}

process OVERLAY_MONTAGE {
    //containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    tag "$well"

    input:
    tuple val(ready),
      val(exp),
      val(morphology_channel),
      val(well),
      val(chosen_timepoints),
      val(wells_toggle),
      val(timepoints_toggle),
      val(channels_toggle),
      val(shift),
      val(contrast)


    output:
    tuple val(well), val(true)

    script:
    """
    overlay_montage.py --experiment ${exp} --target_channel ${morphology_channel} \
    --chosen_wells ${well} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} \
    --channels_toggle ${channels_toggle} --shift ${shift} --contrast ${contrast}
    """
}


process BUNDLED_WORKFLOW_IXM {
    //containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    tag "BUNDLED_WORKFLOW_IXM-${exp}_${well}"
    
    // Resource requirements for the bundled process
    // These can be adjusted based on your cluster capacity
    cpus 20
    memory 64.GB
    time '6h'
    
    // Process-specific resource hints (for monitoring)
    // MONTAGE: ~2 CPU, ~8GB RAM
    // SEGMENTATION: ~4 CPU, ~12GB RAM (most intensive)
    // TRACKING: ~2 CPU, ~6GB RAM
    // OVERLAY: ~1 CPU, ~4GB RAM
    // TOTAL: ~20 CPU, ~64GB RAM (with headroom for parallel processing)

    input:
    tuple val(exp),
          val(tiletype),
          val(montage_pattern),
          val(chosen_timepoints),
          val(chosen_channels),
          val(wells_toggle),
          val(timepoints_toggle),
          val(channels_toggle),
          val(image_overlap),
          val(morphology_channel),
          val(segmentation_method),
          val(img_norm_name),
          val(lower_area_thresh),
          val(upper_area_thresh),
          val(sd_scale_factor),
          val(track_type),
          val(distance_threshold),
          val(target_channel),
          val(well),
          val(shift),
          val(contrast)

    output:
    tuple val(well), val(true)

    script:
    """
    #!/bin/bash
    set -e
    
    # ============================================================================
    # BUNDLED WORKFLOW CONFIGURATION
    # ============================================================================
    # This process combines 4 lightweight operations into a single job per well
    # to reduce job launch overhead from 4x to 1x per well.
    #
    # RESOURCE ALLOCATION (adjust in Nextflow process definition above):
    # - CPUs: 4 per well (configurable)
    # - Memory: 16GB per well (configurable)
    # - Time: 6 hours per well (configurable)
    #
    # PROCESS BREAKDOWN:
    # 1. MONTAGE: ~2 CPU, ~8GB RAM
    # 2. SEGMENTATION: ~4 CPU, ~12GB RAM (most intensive)
    # 3. TRACKING: ~2 CPU, ~6GB RAM  
    # 4. OVERLAY: ~1 CPU, ~4GB RAM
    # ============================================================================
    
    # Record start time
    START_TIME=\$(date +%s)
    START_TIMESTAMP=\$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "🚀 Starting bundled workflow for well ${well}"
    echo "📊 Processing: MONTAGE → SEGMENTATION → TRACKING → OVERLAY"
    echo "💻 Resources: 20 CPUs, 64GB RAM, 6h time limit"
    echo "⏰ Started at: \${START_TIMESTAMP}"
    
    # Step 1: MONTAGE
    echo "🔧 Step 1/4: Creating montage for well ${well}"
    montage.py --experiment ${exp} --tiletype ${tiletype} --montage_pattern ${montage_pattern} \
    --chosen_wells ${well} --chosen_timepoints ${chosen_timepoints} --chosen_channels ${chosen_channels} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} --channels_toggle ${channels_toggle} \
    --image_overlap ${image_overlap}
    
    if [ \$? -eq 0 ]; then
        echo "✅ Montage completed successfully for well ${well}"
    else
        echo "❌ Montage failed for well ${well}"
        exit 1
    fi
    
    # Step 2: SEGMENTATION
    echo "🔬 Step 2/4: Running segmentation for well ${well}"
    segmentation_montage.py --experiment ${exp} --segmentation_method ${segmentation_method} \
    --img_norm_name ${img_norm_name} --lower_area_thresh ${lower_area_thresh} --upper_area_thresh ${upper_area_thresh} \
    --sd_scale_factor ${sd_scale_factor} --chosen_wells ${well} --chosen_channels ${morphology_channel} \
    --chosen_timepoints ${chosen_timepoints} --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
    
    if [ \$? -eq 0 ]; then
        echo "✅ Segmentation completed successfully for well ${well}"
    else
        echo "❌ Segmentation failed for well ${well}"
        exit 1
    fi
    
    # Step 3: TRACKING
    echo "🎯 Step 3/4: Running tracking for well ${well}"
    tracking_montage.py --experiment ${exp} --track_type ${track_type} --max_dist ${distance_threshold} \
    --wells ${well} --target_channel ${target_channel}
    
    if [ \$? -eq 0 ]; then
        echo "✅ Tracking completed successfully for well ${well}"
    else
        echo "❌ Tracking failed for well ${well}"
        exit 1
    fi
    
    # Step 4: OVERLAY (Parallel Processing)
    echo "🎨 Step 4/4: Creating overlay for well ${well} (parallel processing)"
    overlay_montage.py --experiment ${exp} --target_channel ${morphology_channel} \
    --chosen_wells ${well} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} \
    --channels_toggle ${channels_toggle} --shift ${shift} --contrast ${contrast}
    
    if [ \$? -eq 0 ]; then
        echo "✅ Overlay completed successfully for well ${well}"
    else
        echo "❌ Overlay failed for well ${well}"
        exit 1
    fi
    
    # Calculate and display total time
    END_TIME=\$(date +%s)
    END_TIMESTAMP=\$(date '+%Y-%m-%d %H:%M:%S')
    TOTAL_TIME=\$((END_TIME - START_TIME))
    TOTAL_MINUTES=\$((TOTAL_TIME / 60))
    TOTAL_SECONDS=\$((TOTAL_TIME % 60))
    
    echo "🎉 Bundled workflow completed successfully for well ${well}!"
    echo "⏰ Completed at: \${END_TIMESTAMP}"
    echo "⏱️  Total time: \${TOTAL_TIME} seconds (\${TOTAL_MINUTES}m \${TOTAL_SECONDS}s)"
    """
}
