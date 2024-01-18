
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


def _mask_keypoint_out_of_roi(image, kp, erode=10):

    # from matplotlib import pyplot as plt
    # plt.imshow(image)

    # Generate the mask
    mask = image > 0
    # plt.figure()
    # plt.imshow(mask)
    # mask = discErosion(mask.astype('uint8'), 1)
    # plt.figure()
    # plt.imshow(mask)
    mask = discErosion(mask.astype('uint8'), erode)
    # plt.figure()
    # plt.imshow(mask)

    # Remove keypoints within the mask
    # point_map = np.zeros(mask.shape, dtype='uint8')
    # point_map_new = np.zeros(mask.shape, dtype='uint8')
    new_kp = []
    for x in kp:
        p = [int(x[0] + 0.5), int(x[1] + 0.5)]
        # point_map[p[1], p[0]] = 255
        if mask[p[1], p[0]] > 0:
            new_kp.append(x)
            # point_map_new[p[1], p[0]] = 255

    # plt.figure()
    # plt.imshow(discDilation(point_map, 5))
    # plt.figure()
    # plt.imshow(discDilation(point_map_new, 5))
    #
    # plt.show()
    return np.array(new_kp)


def _equalize_shape(im0, im1):
    if im0.shape != im1.shape:

        final_shape = np.max((im0.shape, im1.shape), axis=0)
        print(f'im0.shape = {im0.shape}')
        print(f'im1.shape = {im1.shape}')
        print(f'final_shape = {final_shape}')

        tim0 = np.zeros(final_shape, dtype=im0.dtype)
        tim0[:im0.shape[0], :im0.shape[1]] = im0
        tim1 = np.zeros(final_shape, dtype=im1.dtype)
        tim1[:im1.shape[0], :im1.shape[1]] = im1

        return tim0, tim1
    else:
        return im0, im1


def _sift(
        image,
        reference,
        sift_ocl=None,
        devicetype=None,
        norm_quantiles=None,
        return_keypoints=False,
        auto_mask=None,
        verbose=False
):

    if sift_ocl is None and devicetype is None:
        raise RuntimeError('Either sift_ocl or devicetype need to be supplied')

    if norm_quantiles is not None:
        image = _norm_8bit(image, norm_quantiles, ignore_zeros=auto_mask is not None)
        reference = _norm_8bit(reference, norm_quantiles, ignore_zeros=auto_mask is not None) if type(reference) == np.ndarray else reference

    if verbose:
        print(f'image.shape = {image.shape}')
        print(f'reference.shape = {reference.shape}')

    if type(reference) == np.ndarray:
        image, reference = _equalize_shape(image, reference)

    # Initialize the SIFT
    if sift_ocl is None:
        if verbose:
            print('Initializing SIFT')
        assert devicetype is not None
        sift_ocl = sift.SiftPlan(template=image, devicetype=devicetype)

    if verbose:
        print('Computing keypoints')

    # Compute keypoints
    keypoints_moving = sift_ocl.keypoints(image)
    if verbose:
        print(f'len(keypoints_moving) = {len(keypoints_moving)}')
    if auto_mask is not None:
        keypoints_moving = _mask_keypoint_out_of_roi(image, keypoints_moving, erode=auto_mask)
        if verbose:
            print(f'len(keypoints_moving) = {len(keypoints_moving)}')
    if verbose:
        print(f'type(reference) = {type(reference)}')
    if type(reference) == np.ndarray:
        print(f'reference.dtype = {reference.dtype}')
        print(f'sift_ocl.shape = {sift_ocl.shape}')
        print(f'reference.shape = {reference.shape}')
        keypoints_ref = sift_ocl.keypoints(reference)
        if auto_mask:
            keypoints_ref = _mask_keypoint_out_of_roi(reference, keypoints_ref)
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

    # # FIXME remove
    # kp_im = np.zeros(image.shape, dtype='uint8')
    # from vigra.filters import gaussianSmoothing
    # for kp in keypoints_moving:
    #     print(f'kp = {kp}')
    #     kp_im[int(kp[1]), int(kp[0])] = 255
    # kp_im = gaussianSmoothing(kp_im, 5)
    # kp_im = (kp_im.astype(float) / kp_im.max() * 255).astype('uint8')
    # return (float(offset[0]), float(offset[1])), image, reference, kp_im


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
        device_type='GPU',
        return_bounds=False,
        auto_mask=None,
        max_offset=None,
        xy_range=None,
        invert_nonzero=False,
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
    else:
        im = imread(im_fp)
        if return_bounds:
            bounds = get_bounds(im)
        ref_im = imread(ref_im_fp)

    # TODO Apply the bounds to save computational time! Depending on how much zero padding is in the data, this is more
    #   than substantial!

    if invert_nonzero:
        im = _invert_nonzero(im)
        ref_im = _invert_nonzero(ref_im)

    im = preprocess_slice(im, sigma=sigma, mask_range=mask_range, thresh=thresh)
    ref_im = preprocess_slice(ref_im, sigma=sigma, mask_range=mask_range, thresh=thresh)

    # # FIXME remove
    # offsets, im, ref_im, kp_im = _sift(
    #     im, ref_im,
    #     devicetype=device_type,
    #     norm_quantiles=norm_quantiles,
    #     auto_mask=auto_mask,
    #     verbose=verbose
    # )
    # from h5py import File
    # import os
    # with File(
    #         os.path.join(
    #             '/media/julian/Data/tmp/sift_test/',
    #             os.path.split(im_fp)[1]
    #         ),
    #         mode='w'
    # ) as f:
    #     f.create_dataset('im', data=im, compression='gzip')
    #     f.create_dataset('ref_im', data=ref_im, compression='gzip')
    #     f.create_dataset('kp_im', data=kp_im, compression='gzip')
    offsets = _sift(
        im, ref_im,
        devicetype=device_type,
        norm_quantiles=norm_quantiles,
        auto_mask=auto_mask,
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
