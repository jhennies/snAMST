# snAMST
AMST implementation using Snakemake workflow management

## Installation

The basic AMST functionalities 

    conda create -n snamst-env -c conda-forge tifffile scikit-image=0.17.2 vigra silx[full] pyopencl

To make the GPU accessible

    conda activate snamst-env
    conda install -c conda-forge ocl-icd-system

Adding snakemake

    conda install -c conda-forge -c bioconda snakemake