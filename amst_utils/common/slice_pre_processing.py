
from vigra.filters import gaussianSmoothing
from vigra.filters import discRankOrderFilter


def preprocess_slice(image, thresh=None, sigma=1., mask_range=None, med_diff=0):

    if mask_range is not None:
        assert image.dtype == 'uint8', 'Only implemented for uint8 datasets (for now)'
        image[image < mask_range[0]] = mask_range[0]
        image[image > mask_range[1]] = mask_range[0]
        image = (image - mask_range[0]).astype('float32')
        image = (image / (mask_range[1] - mask_range[0]) * 255).astype('uint8')
    if thresh is not None:
        if type(thresh) not in [list, tuple]:
            thresh = [thresh, thresh]
        if thresh[0] > 0:
            image[image < thresh[0]] = thresh[0]
        if thresh[1] > 0:
            image[image > thresh[1]] = thresh[1]

    med_diff = 10
    if med_diff > 0:
        assert image.dtype == 'uint8'
        med_filt = discRankOrderFilter(image, med_diff, rank=0.5)
        image = image.astype('float32') - med_filt.astype('float32')
        image[image < 0] = 0
        image = image.astype('uint8')
    if sigma > 0:
        dtype = image.dtype
        image = gaussianSmoothing(image.astype('float32'), sigma)
        image = image.astype(dtype)

    return image
