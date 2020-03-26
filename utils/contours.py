from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, gray2rgb, rgb2gray, label2rgb
from skimage.segmentation import slic
import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from perspective_transform import apply_transform

def get_contours(image):
    image_slic = slic(image,n_segments=5,sigma=5)
    image = label2rgb(image_slic, image, kind='avg')
    mask = np.ma.masked_equal(image, np.min(image))
    return mask


if __name__ == '__main__':
    image = io.imread("test/green.png")
    image = apply_transform(image)

    if len(image.shape) < 3:
        image = gray2rgb(image)
    elif image.shape[-1] > 3:
        image = image[:,:,0]

    if np.max(image) > 1:
        image = image/255

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(image, 0.8)
    print("Found {} contours".format(len(contours)))

    # Display the image and plot all contours found
    fig, ax = plt.subplots()
    ax.imshow(image, cmap=plt.cm.gray)

    for n, contour in enumerate(contours):
        ax.plot(contour[:, 1], contour[:, 0], linewidth=2)

    ax.axis('image')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.savefig('test/contours.png')
