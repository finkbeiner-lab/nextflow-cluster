#!/usr/bin/env nextflow
/*
 * pipeline input parameters
 */

params.experiment = '20230404-NSCLC-Ph1-AZD-1'
params.greeting = 'Hello world!'

experiment_ch = Channel.of(params.experiment)
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
    letters_ch = SPLITLETTERS(greeting_ch)
    results_ch = CONVERTTOUPPER(letters_ch.flatten())
    results_ch.view{ it }
    bashresults_ch = BASHEX(experiment_ch)
    bashresults_ch.view{ it }
    pyresults_ch = GETCSVS(experiment_ch)
    pyresults_ch.view{ it }
}

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : "Oops .. something went wrong" )
}
