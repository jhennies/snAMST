
from tifffile import imread
import json


if __name__ == '__main__':

    # Snakemake inputs
    source = snakemake.input[0]
    target = snakemake.output[0]

    # TODO all the below is temporary for testing
    im = imread(source)

    with open(target, mode='w') as f:
        json.dump(dict(offset=[0., 0.]), f, indent=2)
