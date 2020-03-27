from skimage import io
from utils.crop import *
from utils.img import *
from utils.contours import *
from skimage.draw import polygon2mask


def get_object_crop(image_orig):
    image = apply_transform(image_orig)  # geometric transform
    image = normalize_img(image)  # make sure it's RGB & in range (0,1)
    image_value = image[:,:,2]  # select a single channel for contouring

    # Find contours, select best one and get crop coordinates
    contours = get_contours(image_value)
    print("Found {} contours".format(len(contours)))
    crop_coords, segmented_img = get_crop_from_contours(image, contours)

    # reverse transformation
    segmented_img = apply_transform(segmented_img, inverse=True)
    crop_coords = apply_transform(crop_coords, coords=True)
    cropped_img = image_orig[crop_coords[0][1]:crop_coords[1][1], crop_coords[0][0]:crop_coords[1][0]]

    return crop_coords, cropped_img, segmented_img


if __name__ == '__main__':

    # usage example
    image_orig = io.imread("test/green.png")
    crop_coords, cropped_img, segmented_img = get_object_crop(image_orig)


    # plot everything
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(ncols=3, figsize=(10, 3))
    ax[0].imshow(image_orig)
    ax[1].imshow(cropped_img)
    ax[2].imshow(segmented_img)
    for a in ax:
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig('test/crop_plot.png', dpi=500)
