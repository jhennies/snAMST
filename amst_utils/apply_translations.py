
from tifffile import imread, imsave


if __name__ == '__main__':

    # Snakemake inputs
    source = snakemake.input[0]
    target = snakemake.output[0]

    # TODO all the below is temporary for testing
    im = imread(source)
    imsave(target, data=im, compress=9)
