
import os
import json
from tifffile import imread, imsave


if __name__ == '__main__':

    # Snakemake inputs
    im_fp = snakemake.input[0]
    final_offsets_fp = snakemake.input[1]
    output = snakemake.output[0]

    # TODO all the below is temporary for testing
    im_name = os.path.split(im_fp)[1]
    with open(final_offsets_fp, mode='r') as f:
        offset = json.load(f)[im_name]

    print(f'offset = {offset}')

    im = imread(im_fp)
    imsave(output, data=im, compress=9)
