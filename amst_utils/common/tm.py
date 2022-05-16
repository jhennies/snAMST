
from skimage.feature import match_template
from vigra.filters import gaussianSmoothing
import numpy as np
from tifffile import imread
from amst_utils.common.data import get_bounds


def _tm(image, template_im, thresh=(0, 0), sigma=1.):

    if thresh[0] > 0:
        image[image < thresh[0]] = thresh[0]
        template_im[template_im < thresh[0]] = thresh[0]
    if thresh[1] > 0:
        image[image > thresh[1]] = thresh[1]
        template_im[template_im > thresh[1]] = thresh[1]
    if sigma > 0:
        if image.dtype == 'uint16':
            assert template_im.dtype == 'uint16'
            image = gaussianSmoothing(image.astype('float32'), sigma).astype('uint16')
            template_im = gaussianSmoothing(template_im.astype('float32'), sigma).astype('uint16')
        else:
            image = gaussianSmoothing(image, sigma)
            template_im = gaussianSmoothing(template_im, sigma)

    result = match_template(image, template_im)
    ij = np.unravel_index(np.argmax(result), result.shape)
    x, y = ij[::-1]

    return x, y


def offsets_with_tm(im_fp, template_fp, threshold=(0, 0), sigma=0, return_bounds=False):

    im = imread(im_fp)
    template_im = imread(template_fp)

    offsets = _tm(
        im, template_im, thresh=threshold, sigma=sigma
    )
    offsets = -np.array(offsets)

    if return_bounds:
        return offsets.tolist(), get_bounds(im)
    else:
        return offsets.tolist()
