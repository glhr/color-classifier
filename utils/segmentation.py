from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, gray2rgb, rgb2gray, label2rgb
from skimage.segmentation import slic
import numpy as np
import cv2


def get_segmentation_mask(image):
    image_slic = slic(image,n_segments=5,sigma=5)
    image = label2rgb(image_slic, image, kind='avg')
    mask = np.ma.masked_equal(image, np.min(image))
    return mask


if __name__ == '__main__':
    image = io.imread("test/green.png")

    if len(image.shape) < 3:
        image = gray2rgb(image)
    elif image.shape[-1] > 3:
        image = image[:,:,:3]

    mask = get_segmentation_mask(image)
    cv2.namedWindow('img',cv2.WINDOW_NORMAL)
    cv2.imshow('img',image.astype(np.uint8)*255)
    cv2.waitKey(0)
