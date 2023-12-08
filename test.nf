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
    val ready
    val x

    output: 
    stdout 

    script:
    """
    test.py --txt ${x}
    """
}

process generateStrings {

    output:
    stdout

    script:
    """
    #!/bin/bash
    echo "hello"
    # Generate a list of strings with leading zeros for single-digit numbers
    for LETTER in {A..C}; do
        for number in {1..11}; do
            echo "${LETTER}${number}"
        done
    done
    """
}


process BASHEX {
    tag "Bash Script Test"

    input:
    val x

    output:
    val true
    
    script:
    """
    test.sh
    """
}

process BASHEX2 {
    tag "Bash Script Test"

    input:
    val x

    output:
    val true
    
    script:
    """
    test.sh
    """
}
process BASHEX3 {
    tag "Bash Script Test"

    input:
    val x

    output:
    val true
    
    script:
    """
    test.sh
    """
}
workflow {
    // letters_ch = SPLITLETTERS(greeting_ch)
    // results_ch = CONVERTTOUPPER(letters_ch.flatten())
    // results_ch.view{ it }
    BASHEX(text_ch)
    bashresults_ch = BASHEX.out
    bashresults_ch2 = BASHEX2(text_ch)
    // bashresults_ch3 = BASHEX3(text_ch)
    bashresults_ch3 = Channel.of(true)
    // bashresults_ch.view{ it }
    bash_flag = bashresults_ch.mix(bashresults_ch2).mix(bashresults_ch3).collect()
    pyresults_ch = PYEXAMPLE(bash_flag, text_ch)
    pyresults_ch.view{ it }
}

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : "Oops .. something went wrong" )
}
