try:
    from contours import *
except ImportError:
    from utils.contours import *

try:
    from perspective_transform import apply_transform
except ImportError:
    from utils.perspective_transform import apply_transform

from skimage.draw import polygon2mask
import numpy as np

def get_crop_from_contours(image,contours):
    masks = []

    for contour in contours:
        mask = polygon2mask(image[:,:,0].shape, contour)
        masks.append(mask)

    i = np.argmax([np.sum(mask.astype(np.uint8)) for mask in masks])
    mask = masks[i]
    masked = (image.copy()*255).astype(np.uint8)
    masked[~mask,:] = 0
    masked[mask,:] = 255

    boxes = [[
        [int(np.floor(min(contour[:, 1]))), int(np.floor(min(contour[:, 0])))], # top-left point
        [int(np.ceil(max(contour[:, 1]))), int(np.ceil(max(contour[:, 0])))]  # down-right point
      ] for contour in contours]

    crop = boxes[i]
    return crop, masked


if __name__ == '__main__':

    image_orig = io.imread("test/green.png")

    image = apply_transform(image_orig) # geometric transform
    image = normalize_img(image)  # make sure it's RGB in range (0,1)
    image_value = image[:,:,2]  # select a single channel for contouring

    # Find contours, select best one and get crop coordinates
    contours = get_contours(image_value)
    print("Found {} contours".format(len(contours)))
    crop_coords, segmented_img = get_crop_from_contours(image, contours)
    print(crop_coords)
