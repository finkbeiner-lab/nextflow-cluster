#!/usr/bin/env nextflow

params.greeting = 'Hello world!'
greeting_ch = Channel.of(params.greeting)


process REGISTER_EXPERIMENT {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val input_path
    val output_path
    val template_path
    val platemap_path
    val ixm_hts_file
    val robo_num
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
    --platemap_path ${platemap_path} --ixm_hts_file ${ixm_hts_file} --robo_num ${robo_num} \
     --chosen_wells ${chosen_wells} --chosen_channels ${chosen_channels} --chosen_timepoints ${chosen_timepoints} \
     --wells_toggle ${wells_toggle} --channels_toggle ${channels_toggle} --timepoints_toggle ${timepoints_toggle}
    """

}

process SEGMENTATION {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
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

    output: 
    val true

    script:
    """
    segmentation.py --experiment ${exp} --segmentation_method ${segmentation_method} \
    --img_norm_name ${img_norm_name}  --lower_area_thresh ${lower_area_thresh} --upper_area_thresh ${upper_area_thresh} \
    --sd_scale_factor ${sd_scale_factor} \
    --chosen_wells ${chosen_wells} --chosen_channels ${morphology_channel} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
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
    val morphology_channel
    val segmentation_method
    val lower_area_thresh
    val chosen_wells
    val chosen_timepoints
    val wells_toggle
    val timepoints_toggle

    output:
    val true

    script:
    """
    puncta.py --experiment ${exp} --segmentation_method ${segmentation_method} \
    --chosen_channels ${morphology_channel} --target_channel ${target_channel} \
    --area_thresh ${lower_area_thresh} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle}
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

process MONTAGE {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val ready
    val exp
    val tiletype
    val montage_pattern
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
    montage.py --experiment ${exp} --tiletype ${tiletype} --montage_pattern ${montage_pattern} \
    --chosen_channels ${chosen_channels} --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} \
    --wells_toggle ${wells_toggle} --channels_toggle ${channels_toggle} --timepoints_toggle ${timepoints_toggle}
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
    cnn.py --experiment ${exp} --label_type ${label_type} --label_name ${label_name} --classes ${classes} \
    --img_norm_name ${img_norm_name} --filters ${filters} --num_channels ${num_channels} --n_samples ${n_samples} \
    --epochs ${epochs} --batch_size ${batch_size} --learning_rate ${learning_rate} \
    --momentum ${momentum} --optimizer ${optimizer} \
    --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints} --chosen_channels ${chosen_channels}\
    --wells_toggle ${wells_toggle} --timepoints_toggle ${timepoints_toggle} --channels_toggle ${channels_toggle}
    """
}

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

process MULT {
    input:
    val register_result
    val seg_result
    val track_result
    val intensity_result
    val crop_result
    val cnn_result

    output:
    val mult_result

    script:
    """
    mult_result = register_result * seg_result * track_result * intensity_result * crop_result * cnn_result
    echo "Ready to Get CSVS ($mult_result)"
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

