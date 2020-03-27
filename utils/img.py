from skimage.color import gray2rgb
import numpy as np


def normalize_img(image):
    # print(image.shape)
    if len(image.shape) < 3:
        image = gray2rgb(image)
    elif image.shape[-1] > 3:
        image = image[:,:,:3]
    # image = resize(image, (100,100,3))
    if np.max(image) > 1:
        image = image/255
    # print(image.shape)
    return image
