import os.path

from tifffile import imread
import json


if __name__ == '__main__':

    # Snakemake inputs
    inputs = snakemake.input
    output = snakemake.output[0]

    # TODO all the below is temporary for testing
    image_names = []
    offsets = []
    for inp in inputs:
        with open(inp, mode='r') as f:
            in_data = json.load(f)
        image_names.append(os.path.splitext(os.path.split(inp)[1])[0])
        offsets.append(in_data['offset'])

    results = dict(zip(image_names, offsets))

    with open(output, mode='w') as f:
        json.dump(results, f, indent=2)
