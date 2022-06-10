
import numpy as np
import os
from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter1d
from tifffile import imread, imwrite
from scipy.ndimage.interpolation import shift


def bounds2slice(bounds):

    bounds = np.array(bounds).astype(int)

    return np.s_[
        bounds[0]: bounds[2],
        bounds[1]: bounds[3]
    ]


def smooth_offsets(offsets, median_radius=0, gaussian_sigma=0., suppress_x=False):

    disps_y = offsets[:, 1]
    if median_radius > 0:
        disps_y = medfilt(disps_y, median_radius * 2 + 1)
    if gaussian_sigma > 0:
        disps_y = gaussian_filter1d(disps_y, gaussian_sigma)

    if not suppress_x:

        disps_x = offsets[:, 0]
        if median_radius > 0:
            disps_x = medfilt(disps_x, median_radius * 2 + 1)
        if gaussian_sigma > 0:
            disps_x = gaussian_filter1d(disps_x, gaussian_sigma)

    else:
        disps_x = np.zeros(disps_y.shape, dtype=disps_y.dtype)

    return np.concatenate([disps_x[:, None], disps_y[:, None]], axis=1)


def compute_auto_pad(offsets, bounds):
    """
    :param offsets:
    :param bounds: format: [y, x, y+h, x+w]
    :return:
    """

    # Prepare the displacements respective all bounds and figure out the target shape
    offsets_ = []
    for idx, b in enumerate(bounds):
        offsets_.append(np.array((offsets[idx][0] + b[1], offsets[idx][1] + b[0])))
    offsets = offsets_
    offsets = offsets - np.min(offsets, axis=0)
    starts = []
    stops = []
    for idx, b in enumerate(bounds):
        starts.append(offsets[idx][::-1])
        stops.append(np.array((b[2], b[3])) - np.array((b[0], b[1])) + offsets[idx][::-1])
    min_yx = np.floor(np.min(starts, axis=0)).astype(int)
    max_yx = np.ceil(np.max(stops, axis=0)).astype(int)
    # Pad a little to each side to make it less squished
    # FIXME make this available as a parameter (amount of padding, e.g. pad_result)
    shape = max_yx - min_yx  # + 32
    # offsets += 16

    return offsets, shape


def displace_slice(
        target_fp,
        im_filepath,
        displacement,
        subpx_displacement=False,
        compression=None,
        pad_zeros=None,
        bounds=np.s_[:],
        target_shape=None
):

    # displacement = np.array((
    #         displacement[0] - bounds[1].start,
    #         displacement[1] - bounds[0].start
    # ))

    # Load image
    im = imread(im_filepath)[bounds]
    if target_shape is not None:
        tim = np.zeros(target_shape, im.dtype)
        tim[:im.shape[0], :im.shape[1]] = im
        im = tim

    # zero-pad image
    if pad_zeros:
        pad_im = np.zeros(np.array(im.shape) + (2*pad_zeros), dtype=im.dtype)
        pad_im[pad_zeros: -pad_zeros, pad_zeros: -pad_zeros] = im
        im = pad_im

    print(f'displacement = {displacement}')

    # Shift the image
    if not subpx_displacement:
        im = shift(im, (np.round(displacement[1]), np.round(displacement[0])))
    else:
        im = shift(im, (displacement[1], displacement[0]))

    # Write result
    imwrite(target_fp, im.astype(im.dtype), compression=compression)


def sequentialize_offsets(offsets):

    seq_offset = offsets[0]
    seq_offsets = [seq_offset.copy()]

    for offset in offsets[1:]:
        seq_offset += np.array(offset)
        seq_offsets.append(seq_offset.copy())

    return np.array(seq_offsets)

