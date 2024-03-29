
from tifffile import imread
import numpy as np


def _crop_zero_padding(dat, return_ints=False):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(dat)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)

    if return_ints:
        return np.array([top_left[0], top_left[1], bottom_right[0] + 1, bottom_right[1] + 1]).astype(float).tolist()
    else:
        # generate bounds
        bounds = np.s_[top_left[0]:bottom_right[0] + 1,  # plus 1 because slice isn't
                       top_left[1]:bottom_right[1] + 1]  # inclusive
        return bounds


def get_bounds(im):
    if type(im) == str:
        im = imread(im)
    return _crop_zero_padding(im, return_ints=True)
