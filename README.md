# snAMST
AMST implementation using Snakemake workflow management

## Installation

The basic AMST functionalities 

    conda create -n snamst-env -c conda-forge tifffile scikit-image=0.17.2 vigra silx[full] pyopencl
    conda activate snamst-env

To make the GPU accessible

    conda install -c conda-forge ocl-icd-system

Alternatively, for CPU computation of the SIFT

    conda install -c conda-forge ocl-pocl

Adding snakemake

    conda install -c conda-forge -c bioconda snakemake