process hello {
    container = '/gladstone/finkbeiner/steve/work/projects/datastudy/datastudy-cluster.sif'
    containerEngine = 'singularity'

    script:
    """
    echo "Hello from Singularity"
    """
}

workflow {
    hello()
}

