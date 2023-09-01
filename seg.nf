#!/usr/bin/env nextflow
/*
 * pipeline input parameters
 */

params.experiment = '20230807-KS1-neuron-optocrispr'
params.segmentation_method = 'sd_from_mean'
params.img_norm_name = 'subtraction'
params.lower_area_thresh = 50
params.upper_area_thresh = 2500
params.sd_scale_factor = 3.5
params.chosen_wells = 'E7'
params.chosen_timepoints = 'T0'
params.greeting = 'Hello world!'

experiment_ch = Channel.of(params.experiment)
seg_ch = Channel.of(params.segmentation_method)
norm_ch = Channel.of(params.img_norm_name)
lower_ch = Channel.of(params.lower_area_thresh)
upper_ch = Channel.of(params.upper_area_thresh)
sd_ch = Channel.of(params.sd_scale_factor)
well_ch = Channel.of(params.chosen_wells)
tp_ch = Channel.of(params.chosen_timepoints)
greeting_ch = Channel.of(params.greeting)

include { SPLITLETTERS; CONVERTTOUPPER } from './modules.nf'



params.outdir = "results"

log.info """\
    P Y T H O N   H E L L O  W O R L D !
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
    letters_ch = SPLITLETTERS(greeting_ch)
    results_ch = CONVERTTOUPPER(letters_ch.flatten())
    results_ch.view{ it }
    bashresults_ch = BASHEX(experiment_ch)
    bashresults_ch.view{ it }
    pyresults_ch = SEGMENTATION(experiment_ch, seg_ch, norm_ch, lower_ch, upper_ch, sd_ch, well_ch, tp_ch)
    pyresults_ch.view{ it }
}

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : "Oops .. something went wrong" )
}
