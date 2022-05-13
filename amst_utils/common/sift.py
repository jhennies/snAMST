
from tifffile import imread
import numpy as np
from silx.image import sift
from .data import get_bounds

from amst_utils.common.slice_pre_processing import preprocess_slice


def _norm_8bit(im, quantiles):
    im = im.astype('float32')
    upper = np.quantile(im, quantiles[1])
    lower = np.quantile(im, quantiles[0])
    im -= lower
    im /= (upper - lower)
    im *= 255
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


def _sift(
        image,
        reference,
        sift_ocl=None,
        devicetype=None,
        norm_quantiles=None,
        return_keypoints=False,
        verbose=False
):

    if sift_ocl is None and devicetype is None:
        raise RuntimeError('Either sift_ocl or devicetype need to be supplied')

    if norm_quantiles is not None:
        image = _norm_8bit(image, norm_quantiles)
        reference = _norm_8bit(reference, norm_quantiles) if type(reference) == np.array else reference

    # Initialize the SIFT
    if sift_ocl is None:
        if verbose:
            print('Initializing SIFT')
        assert devicetype is not None
        sift_ocl = sift.SiftPlan(template=image, devicetype=devicetype)

    if verbose:
        print('Computing keypoints')

    # Compute keypoints
    keypoints_moving = sift_ocl(image)
    if verbose:
        print(f'type(reference) = {type(reference)}')
    if type(reference) == np.ndarray:
        keypoints_ref = sift_ocl(reference)
    else:
        if reference is None:
            return (0., 0.), keypoints_moving
        keypoints_ref = reference

    if verbose:
        print('Matching keypoints')

    # Match keypoints
    mp = sift.MatchPlan()
    match = mp(keypoints_ref, keypoints_moving)

    if verbose:
        print('Computing offset')

    # Determine offset
    if len(match) == 0:
        print('Warning: No matching keypoints found!')
        offset = (0., 0.)
    else:
        offset = (np.median(match[:, 1].x - match[:, 0].x), np.median(match[:, 1].y - match[:, 0].y))

    if return_keypoints:
        return (float(offset[0]), float(offset[1])), keypoints_moving
    else:
        return float(offset[0]), float(offset[1])


def offset_with_sift(
        im_fp, ref_im_fp,
        mask_range=None,
        sigma=1.6,
        norm_quantiles=(0.1, 0.9),
        device_type='GPU',
        return_bounds=False,
        verbose=False
):

    im = imread(im_fp)
    ref_im = imread(ref_im_fp)

    im = preprocess_slice(im, sigma=sigma, mask_range=mask_range)
    ref_im = preprocess_slice(ref_im, sigma=sigma, mask_range=mask_range)

    offsets = _sift(
        im, ref_im,
        devicetype=device_type,
        norm_quantiles=norm_quantiles,
        verbose=verbose
    )
    offsets = -np.array(offsets)

    if return_bounds:
        return offsets.tolist(), get_bounds(im)
    else:
        return offsets.tolist()
