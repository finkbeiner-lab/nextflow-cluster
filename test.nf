#!/usr/bin/env nextflow
/*
 * pipeline input parameters
 */

params.text = 'Testing!'
params.greeting = 'Hello world!'

text_ch = Channel.of(params.text)
greeting_ch = Channel.of(params.greeting)

include { SPLITLETTERS; CONVERTTOUPPER } from './modules.nf'


params.outdir = "results"

log.info """\
    P Y T H O N   H E L L O  W O R L D !
    ===================================
    python input : ${params.text}
    greeting     : ${params.greeting}
    outdir       : ${params.outdir}
    """
    .stripIndent()

/*
 * define the `index` process that creates a binary index
 * given the transcriptome file
 */

process PYEXAMPLE {
    input:
    val x

    output: 
    stdout 

    script:
    """
    test.py --txt ${x}
    """
}

process BASHEX {
    tag "Bash Script Test"

    input:
    val x

    output:
    stdout
    
    script:
    """
    test.sh
    """
}

workflow {
    letters_ch = SPLITLETTERS(greeting_ch)
    results_ch = CONVERTTOUPPER(letters_ch.flatten())
    results_ch.view{ it }
    bashresults_ch = BASHEX(text_ch)
    bashresults_ch.view{ it }
    pyresults_ch = PYEXAMPLE(text_ch)
    pyresults_ch.view{ it }
}

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : "Oops .. something went wrong" )
}
