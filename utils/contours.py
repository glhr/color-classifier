from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, gray2rgb, rgb2gray, label2rgb
from skimage.segmentation import slic
import cv2

import numpy as np


from skimage import measure
try:
    from perspective_transform import apply_transform
    from img import normalize_img
except ImportError:
    from utils.perspective_transform import apply_transform
    from utils.img import normalize_img
import scipy.ndimage as ndimage
from skimage.draw import polygon2mask


def get_contours(image):
    return measure.find_contours(image, 0.8)


def get_masks_from_contours(image, contours):
    masks = []
    for contour in contours:
        # # Create an empty image to store the masked array
        # mask = np.zeros_like(image, dtype='bool')
        # # Create a contour image by using the contour coordinates rounded to their nearest integer value
        # mask[np.round(contour[:, 0]).astype('int'), np.round(contour[:, 1]).astype('int')] = 1
        # # Fill in the hole created by the contour boundary
        # mask = ndimage.binary_fill_holes(mask)
        # # Invert the mask since you want pixels outside of the region
        # # mask = ~mask
        mask = polygon2mask(image.shape, contour)
        masks.append(mask)
    return masks


def get_masked_image(image, masks):
    # get largest mask
    i = np.argmax([np.sum(mask.astype(np.uint8)) for mask in masks])

    if np.sum(masks[i].astype(np.uint8)) < 2000:
        return None
    mask = masks[i]
    masked = (image.copy()*255).astype(np.uint8)
    masked[~mask,:] = 255

    return masked, mask


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    
    image = io.imread("test/green.png")
    image = apply_transform(image)

    image = normalize_img(image)

    image_value = image[:,:,2]

    # Find contours at a constant value of 0.8
    contours = get_contours(image_value)
    print("Found {} contours".format(len(contours)))

    # Display the image and plot all contours found
    fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
    ax[0].imshow(image)
    for n, contour in enumerate(contours):
        ax[0].plot(contour[:, 1], contour[:, 0], linewidth=2)

    masks = get_masks_from_contours(image_value, contours)

    masked, mask = get_masked_image(image, masks)

    io.imsave("test/mask.png", masked)
    ax[1].imshow(masked)
    for a in ax:
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])

    plt.tight_layout()
    plt.savefig('test/contours_plot.png', dpi=500)
