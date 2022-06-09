
from tifffile import imread
import numpy as np
from silx.image import sift
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


def _mask_keypoint_out_of_roi(image, kp, erode=10):

    from matplotlib import pyplot as plt
    from vigra.filters import discDilation, discErosion
    plt.imshow(image)

    # Generate the mask
    mask = image > 0
    # plt.figure()
    # plt.imshow(mask)
    # mask = discErosion(mask.astype('uint8'), 1)
    # plt.figure()
    # plt.imshow(mask)
    mask = discErosion(mask.astype('uint8'), erode)
    plt.figure()
    plt.imshow(mask)

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


def offset_with_sift(
        im_fp, ref_im_fp,
        mask_range=None,
        thresh=None,
        sigma=1.6,
        norm_quantiles=(0.1, 0.9),
        device_type='GPU',
        return_bounds=False,
        auto_mask=None,
        verbose=False
):

    im = imread(im_fp)
    ref_im = imread(ref_im_fp)

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

    if return_bounds:
        return offsets.tolist(), get_bounds(im)
    else:
        return offsets.tolist()
