#!/usr/bin/env nextflow
/*
 * pipeline input parameters
 https://nextflow-io.github.io/patterns/conditional-process/
 */
// Global variables
params.experiment = '20230828-2-msneuron-cry2'
params.csvdir = '/gladstone/finkbeiner/robodata/ThinkingMicroscope-DB/GXYTMP_20230828-2-msneuron-cry2/CSVS'

// Variables per module

// DELETE EXPERIMENT
params.DO_DELETE = true

// COPY CSVS TO DATABASE
params.DO_COPY = true


experiment_ch = Channel.of(params.experiment)
csvdir_ch = Channel.of(params.csvdir)
greeting_ch = Channel.of(params.greeting)

include { SPLITLETTERS; CONVERTTOUPPER } from './modules.nf'



params.outdir = "results"

log.info """\
    COPY DATABASE TABLES FROM CSV
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


process DELETEEXP {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val exp

    output:
    val true

    script:
    """
    delete_experiment.py --experiment ${exp}
    """
}

process COPYTODB {
    containerOptions "--mount type=bind,src=/gladstone/finkbeiner/,target=/gladstone/finkbeiner/"
    input:
    val ready
    val exp
    val csvdir

    output:
    stdout

    script:
    """
    copy_db.py --experiment ${exp} --csvdir ${csvdir}
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
    if ( params.DO_DELETE ) {
        del_ch = DELETEEXP(experiment_ch)
    }
    if ( params.DO_COPY){
        copydb_ch = COPYTODB(DELETEEXP.out, experiment_ch, csvdir_ch)
    }
}

workflow.onComplete {
    log.info ( workflow.success ? "\nDone! Open the following report in your browser --> $params.outdir/pyexample_report.html\n" : "Oops .. something went wrong" )
}
