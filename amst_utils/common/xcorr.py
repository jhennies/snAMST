
from skimage.registration import phase_cross_correlation
from vigra.filters import gaussianSmoothing
from skimage import filters
from tifffile import imread
import numpy as np
from vigra.filters import discErosion

from .data import get_bounds
from amst_utils.common.slice_pre_processing import preprocess_slice


def _norm_8bit(im, quantiles, ignore_zeros=False):
    im = im.astype('float32')
    if ignore_zeros:
        upper = np.quantile(im[im > 0], quantiles[1])
        lower = np.quantile(im[im > 0], quantiles[0])
    else:
        upper = np.quantile(im, quantiles[1])
        lower = np.quantile(im, quantiles[0])
    im -= lower
    im /= (upper - lower)
    im *= 255
    im[im > 255] = 255
    im[im < 0] = 0
    return im.astype('uint8')


def _generate_mask(image, erode=10):

    mask = image > 0
    mask = discErosion(mask.astype('uint8'), erode)

    return mask


def _xcorr(
        image,
        reference,
        norm_quantiles=None,
        auto_mask=None,
        verbose=False
):

    image = filters.sobel(image)
    reference = filters.sobel(reference)

    if norm_quantiles is not None:
        image = _norm_8bit(image, norm_quantiles, ignore_zeros=auto_mask is not None)
        reference = _norm_8bit(reference, norm_quantiles, ignore_zeros=auto_mask is not None) if type(reference) == np.ndarray else reference

    shift = phase_cross_correlation(
        reference, image,
        reference_mask=_generate_mask(reference, auto_mask) if auto_mask is not None else None,
        moving_mask=_generate_mask(image, auto_mask) if auto_mask is not None else None,
        upsample_factor=10
    )
    return shift[1], shift[0]


def offsets_with_xcorr(
        im_fp, ref_im_fp,
        mask_range=None,
        thresh=None,
        sigma=1.,
        norm_quantiles=(0.1, 0.9),
        return_bounds=False,
        auto_mask=None,
        verbose=False
):

    im = imread(im_fp)
    ref_im = imread(ref_im_fp)

    im = preprocess_slice(im, sigma=sigma, mask_range=mask_range, thresh=thresh)
    ref_im = preprocess_slice(ref_im, sigma=sigma, mask_range=mask_range, thresh=thresh)

    offsets = _xcorr(
        im, ref_im,
        norm_quantiles=norm_quantiles,
        auto_mask=auto_mask,
        verbose=verbose
    )
    offsets = -np.array(offsets)

    if return_bounds:
        return offsets.tolist(), get_bounds(im)
    else:
        return offsets.tolist()

