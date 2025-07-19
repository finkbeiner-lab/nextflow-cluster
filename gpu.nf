nextflow.enable.dsl=2

process gpu_test {
    container = '/gladstone/finkbeiner/steve/work/projects/datastudy/datastudy-cluster.sif'
    containerEngine = 'singularity'
    label 'gpu'
    
    cpus = 1
    memory = '32 GB'
    time = '2h'
    clusterOptions = '--gres=gpu:v100:1'

    output:
    path 'gpu_test_output.txt'

    script:
    """
    echo "Inside container, node: \$(hostname)"
    nvidia-smi || echo "No GPU found"
    python /app/test_gpu.py > gpu_test_output.txt
    cat gpu_test_output.txt
    """
}

workflow {
    gpu_test()
}

