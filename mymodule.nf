#!/usr/bin/env nextflow
params.greeting = "Hello Everybody"

greeting_channel = Channel.of(params.greeting)

process Test{
    input:

    output:
    stdout

    script:
    """
    mymodule.py
    """

}

workflow{
    test_channel = Test()
    test_channel.view()
}