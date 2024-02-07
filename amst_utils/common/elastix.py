
from tifffile import imread
import numpy as np
from .data import get_bounds
from vigra.filters import discErosion

from amst_utils.common.slice_pre_processing import preprocess_slice
from squirrel.library.elastix import register_with_elastix


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


def _elastix(
        image,
        reference,
        norm_quantiles=None,
        auto_mask=None,
        mask=None,
        downsample=1,
        verbose=False
):
    import cv2 as cv
    if reference is None:
        return (0., 0.)

    if auto_mask is not None:
        assert mask is None, "Don't supply a mask when using auto-masking!"
        mask = image > 0
        if auto_mask > 0:
            mask = discErosion(mask.astype('uint8'), auto_mask)

    if downsample > 1:
        image = cv.resize(image, (np.array(image.shape) / downsample).astype(int), interpolation=cv.INTER_LINEAR)
        reference = cv.resize(reference, (np.array(reference.shape) / downsample).astype(int), interpolation=cv.INTER_LINEAR)
        if mask is not None:
            mask = cv.resize(mask, (np.array(mask.shape) / downsample).astype(int), interpolation=cv.INTER_LINEAR)

    if norm_quantiles is not None:
        if mask is not None:
            im_in = image[:]
            im_in[mask == 0] = 0
            ref_in = reference[:]
            ref_in[mask == 0] = 0
        else:
            im_in = image
            ref_in = reference
        image = _norm_8bit(im_in, norm_quantiles, ignore_zeros=auto_mask is not None)
        reference = _norm_8bit(ref_in, norm_quantiles, ignore_zeros=auto_mask is not None) if type(reference) == np.ndarray else reference

    out_dict = register_with_elastix(
        reference, image,
        transform='translation',
        verbose=verbose
    )

    offset = out_dict['translation_parameters']

    return float(offset[0]), float(offset[1])


def _invert_nonzero(img):
    img_max = img.max()
    img[img > 0] = img_max - img[img > 0]
    return img


def offset_with_elastix(
        im_fp, ref_im_fp,
        mask_range=None,
        thresh=None,
        sigma=1.6,
        norm_quantiles=(0.1, 0.9),
        return_bounds=False,
        auto_mask=None,
        max_offset=None,
        xy_range=None,
        invert_nonzero=False,
        mask_im_fp=None,
        downsample=1,
        bias=(0., 0.),
        verbose=False
):

    bounds = None
    if xy_range is not None:
        x, y, w, h = xy_range
        im = imread(im_fp)
        if return_bounds:
            bounds = get_bounds(im)
        im = im[y:y+h, x:x+w]
        ref_im = imread(ref_im_fp)[y:y+h, x:x+w]
        mask_im = imread(mask_im_fp)[y:y+h, x:x+w] if mask_im_fp is not None else None
    else:
        im = imread(im_fp)
        if return_bounds:
            bounds = get_bounds(im)
        ref_im = imread(ref_im_fp)
        mask_im = imread(mask_im_fp) if mask_im_fp is not None else None

    # TODO Apply the bounds to save computational time! Depending on how much zero padding is in the data, this is more
    #   than substantial!

    if invert_nonzero:
        im = _invert_nonzero(im)
        ref_im = _invert_nonzero(ref_im)

    im = preprocess_slice(im, sigma=sigma, mask_range=mask_range, thresh=thresh)
    ref_im = preprocess_slice(ref_im, sigma=sigma, mask_range=mask_range, thresh=thresh)

    offsets = _elastix(
        im, ref_im,
        norm_quantiles=norm_quantiles,
        auto_mask=auto_mask,
        mask=mask_im,
        downsample=downsample,
        verbose=verbose
    )
    offsets = -np.array(offsets)
    if max_offset is not None:
        if abs(offsets[0]) > max_offset[0] or abs(offsets[1]) > max_offset[1]:
            offsets = np.array([0., 0.])

    # Apply bias
    offsets = offsets + bias

    if return_bounds:
        return offsets.tolist(), bounds
    else:
        return offsets.tolist()
