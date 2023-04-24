# snAMST
AMST implementation using Snakemake workflow management. For the origninal implementation see github.com/jhennies/amst

Note that this repository is a work in progress. Currently only a pre-alignment procedure is implemented.

## Installation

The basic AMST functionalities 

    conda create -n snamst-env -c conda-forge tifffile scikit-image=0.17.2 vigra silx[full] pyopencl
    conda activate snamst-env

To make the GPU accessible

    conda install -c conda-forge ocl-icd-system

Alternatively, for CPU computation of the SIFT

    conda install -c conda-forge pocl

Adding snakemake

    conda install -c conda-forge -c bioconda snakemake

## Usage

Currently only the pre-alignment procedure is implemented. For a usage example see ```example_pre_align.sh```. 

The example shows how to use the pre_align.py to perform a SIFT alignment. For a more advanced alignment procedure including a template matching on a fiducial include the ```--template``` or ```-tm``` argument.

### On the cluster (description for EMBL internal use)

For the following description, change the path names as required.

Log in to a cluster node.

Your input data should sit on the scratch:

    mkdir /scratch/my_name/
    cp -r /g/share_name/path/to/my/dataset_name /scratch/my_name/

Copy the script ```example_pre_align.sh``` to a folder where you keep your run scripts or your results folder
and give it a more specific name. 
For example using:

    cp example_pre_align.sh /path/to/my/scripts/pre_align_dataset_name.sh

Now edit the copied script with the editor or your choice, for example

    cd /path/to/my/scripts
    vim example_pre_align.sh

such that it points to the correct locations:

    #############################################################
    # PARAMETERS
    
    # Data
    source_folder=/scratch/my_name/dataset_name
    target_folder=/scratch/my_name/dataset_aligned
    
    # Workflow parameters
    local_auto_mask=10      # Automatically remove SIFT key-points which are at the edge of the actual (non-zero) data

Increase the number of cores if you are impatient ;) Currently it is way more efficient to use many cores on the HTC as
rather than using GPU support and lesser threads

    # Compute settings
    local_device_type=CPU   # CPU is currently more efficient for the cluster
    cores=128               # Set to 0 for running locally with all cores, otherwise specify a value > 0
    gpus=1                  # Has no effect if local_device_type=CPU
    cluster=slurm           # Can be "none" or "slurm"

The installation settings should point to the correct locations

    # Installation settings (normally don't change anything here)
    conda_path=/g/icem/hennies/software/miniconda3
    env_path=/g/icem/hennies/envs
    src_path=/g/icem/hennies/src/github/jhennies/snamst
    
    #############################################################

When done save and exit the editor.

Now just start the script with pre_align_dataset_name.sh. The entire cluster support is handled automatically. 

When done the results can be copied back to your group share 

    cp -r /scratch/my_name/dataset_aligned /g/share_name/path/to/my/project

## The output

The results folder contains three items:

 - ```log```: can be deleted if the run was successful, contains the log files for each snakemake task that ran on the cluster
 - ```pre_align```: the final results, aka the aligned dataset
 - ```pre_align_cache```: see below

The ```pre_align_cache``` folder will contain three items:

 - A folder called ```offsets_local``` which contains the relative movement for each slice with respect to the previous
 - A file called ```final_offsets.json``` which contains the offsets that need to be performed for the final alignment
 - A file called ```params.json``` which contains all parameters that were used (including defaults)

It can make sense to keep the ```*.json``` files as they describe the run and can be used to reproduce the result
