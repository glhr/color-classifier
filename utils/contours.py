from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, gray2rgb, rgb2gray, label2rgb
from skimage.segmentation import slic
import cv2

import numpy as np
import matplotlib.pyplot as plt

from skimage import measure
from perspective_transform import apply_transform
import scipy.ndimage as ndimage


def get_contours(image):
    image_slic = slic(image, n_segments=5, sigma=5)
    image = label2rgb(image_slic, image, kind='avg')
    mask = np.ma.masked_equal(image, np.min(image))
    return mask


def get_masks_from_contours(image, contours):
    masks = []
    for contour in contours:
        # Create an empty image to store the masked array
        mask = np.zeros_like(image, dtype='bool')
        # Create a contour image by using the contour coordinates rounded to their nearest integer value
        mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
        # Fill in the hole created by the contour boundary
        mask = ndimage.binary_fill_holes(mask)
        # Invert the mask since you want pixels outside of the region
        # mask = ~mask
        masks.append(mask)
    return masks


if __name__ == '__main__':
    image = io.imread("test/green.png")
    image = apply_transform(image)

    if len(image.shape) < 3:
        image = gray2rgb(image)
    elif image.shape[-1] > 3:
        image = image[:,:,:3]

    if np.max(image) > 1:
        image = image/255

    image_value = image[:,:,2]

    # Find contours at a constant value of 0.8
    contours = measure.find_contours(image_value, 0.8)
    print("Found {} contours".format(len(contours)))

    # Display the image and plot all contours found
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    ax[0].imshow(image)
    for n, contour in enumerate(contours):
        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2)

    masks = get_masks_from_contours(image_value, contours)

    for i, mask in enumerate(masks):
        if np.sum(mask.astype(np.uint8)) > 500:
            # broadcast `mask` along the 3rd dimension to make it the same shape as `x`
            masked = (image.copy()*255).astype(np.uint8)
            masked[~mask,:] = 255

            io.imsave("test/mask{}.png".format(i), masked)

    for a in ax:
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.savefig('test/contours_plot.png', dpi=500)
