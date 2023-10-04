
from tifffile import imread
import numpy as np
from silx.image import sift
from .data import get_bounds
from vigra.filters import discErosion

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


def _sift(
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

    # Initialize the SIFT
    sift = cv.SIFT_create()

    if verbose:
        print('Computing keypoints')

    # Compute keypoints
    kp_img, des_img = sift.detectAndCompute(image, mask)
    kp_ref, des_ref = sift.detectAndCompute(reference, mask)

    assert not auto_mask, 'Auto-mask is not implemented right now'

    # Compute matches
    if verbose:
        print('Matching keypoints')
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    # matches = flann.knnMatch(des1, des2, k=2)
    matches = flann.match(des_img, des_ref)

    # Make sure only to use the good matches
    good = []
    # for m, n in matches:
    #     if m.distance < 0.7 * n.distance:
    #         good.append(m)
    matches = sorted(matches, key=lambda x: x.distance)
    good = matches[:100]

    # Only use inliers
    src_pts = np.float32([kp_img[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp_ref[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    M, matches_mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matches_mask = matches_mask.ravel().tolist()

    # Determine offset
    final = np.array(good)[np.array(matches_mask) > 0]
    offset = np.median([
        [kp_img[x.queryIdx].pt[0] - kp_ref[x.trainIdx].pt[0],
         kp_img[x.queryIdx].pt[1] - kp_ref[x.trainIdx].pt[1]]
        for x in final
    ], axis=0)

    if downsample > 1:
        offset = tuple(np.array(offset) * downsample)

    return float(offset[0]), float(offset[1])


def _invert_nonzero(img):
    img_max = img.max()
    img[img > 0] = img_max - img[img > 0]
    return img


def offset_with_sift(
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

    offsets = _sift(
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

    if return_bounds:
        return offsets.tolist(), bounds
    else:
        return offsets.tolist()
