#!/usr/bin/env nextflow
/*
 * pipeline input parameters
 https://nextflow-io.github.io/patterns/conditional-process/
 */
// Global variables
params.chosen_wells = 'all'
params.chosen_timepoints = 'all'
params.experiment = '20230828-2-msneuron-cry2'

// Variables per module

// REGISTER_EXPERIMENT
params.DO_REGISTER_EXPERIMENT = false
params.input_path = ''
params.output_path = ''
params.ixm_hts_file = ''
params.illumination_file = '/gladstone/finkbeiner/robodata/IXM Documents/illumination-setting-2023-06-16.ILS'  // Path to IXM Illumination file. On metaxpres -> Control -> Devices -> Configure Illumination -> Backup
params.robo_num = 0

// SEGMENTATION
params.DO_SEGMENTATION = false
params.segmentation_method = 'sd_from_mean'
params.img_norm_name = 'subtraction'
params.lower_area_thresh = 50
params.upper_area_thresh = 2500
params.sd_scale_factor = 3.5

params.greeting = 'Hello world!'

// TRACKING
params.DO_TRACKING = false
params.distance_threshold = 300 // distance that a cell must be new
params.voronoi_bool = true // distance that a cell must be new

// INTENSITY
params.DO_INTENSITY = true
params.img_norm_for_intensity = 'subtraction'  // ['division', 'subtraction', 'identity']
params.target_channel = ['RFP1','YFP1', 'YFP2', 'RFP2']
params.morphology_channel = 'RFP1'

// GET CSVS
params.DO_GET_CSVS = false



experiment_ch = Channel.of(params.experiment)
seg_ch = Channel.of(params.segmentation_method)
norm_ch = Channel.of(params.img_norm_name)
lower_ch = Channel.of(params.lower_area_thresh)
upper_ch = Channel.of(params.upper_area_thresh)
sd_ch = Channel.of(params.sd_scale_factor)
well_ch = Channel.of(params.chosen_wells)
tp_ch = Channel.of(params.chosen_timepoints)
target_channel_ch = Channel.from(params.target_channel)
morphology_ch = Channel.of(params.morphology_channel)
greeting_ch = Channel.of(params.greeting)

include { SPLITLETTERS; CONVERTTOUPPER } from './modules.nf'



params.outdir = "results"

log.info """\
    ANALYSIS DATASTUDY PIPELINE!
    ===================================
    python input : ${params.experiment}
    greeting     : ${params.greeting}
    outdir       : ${params.outdir}
    """
    .stripIndent()

/*
 * define the `index` process that creates a binary index
 * given the transcriptome file
 */

process SEGMENTATION {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val exp
    val segmentation_method
    val img_norm_name
    val lower_area_thresh
    val upper_area_thresh
    val sd_scale_factor
    val chosen_wells
    val chosen_timepoints

    output: 
    stdout 

    script:
    """
    segmentation.py --experiment ${exp} --segmentation_method ${segmentation_method} --img_norm_name ${img_norm_name}  --lower_area_thresh ${lower_area_thresh} --upper_area_thresh ${upper_area_thresh} --sd_scale_factor ${sd_scale_factor}  --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints}
    """
}

process INTENSITY {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val exp
    val img_norm_name
    val morphology_channel
    each target_channel
    val chosen_wells
    val chosen_timepoints

    output:
    stdout

    script:
    """
    intensity.py --experiment ${exp} --img_norm_name ${img_norm_name}  --chosen_channels ${morphology_channel} --target_channel ${target_channel} --chosen_wells ${chosen_wells} --chosen_timepoints ${chosen_timepoints}
    """
}

process GETCSVS {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val exp

    output:
    stdout

    script:
    """
    get_csvs.py --experiment ${exp}
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

workflow {
    bashresults_ch = BASHEX(experiment_ch)
    bashresults_ch.view{ it }
    if ( params.DO_SEGMENTATION ) {
        seg_ch = SEGMENTATION(experiment_ch, seg_ch, norm_ch, lower_ch, upper_ch, sd_ch, well_ch, tp_ch)
        seg_ch.view{ it }
    }
    if ( params.DO_INTENSITY ) {
        seg_ch = INTENSITY(experiment_ch, norm_ch, morphology_ch, target_channel_ch, well_ch, tp_ch)
        seg_ch.view{ it }
    }
    if ( params.DO_GET_CSVS){
        csv_ch = GETCSVS(experiment_ch)
        csv_ch.view{ it }
    }
}

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : "Oops .. something went wrong" )
}
