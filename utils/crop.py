try:
    from contours import *
except ImportError:
    from utils.contours import *

try:
    from perspective_transform import apply_transform, apply_inverse_transform
except ImportError:
    from utils.perspective_transform import apply_transform, apply_inverse_transform


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

    print(crop)



    return crop, masked


if __name__ == '__main__':
    image_orig = io.imread("test/green.png")
    image = apply_transform(image_orig)
    image = normalize_img(image)
    image_value = image[:,:,2]
    # Find contours at a constant value of 0.8
    contours = get_contours(image_value)
    print("Found {} contours".format(len(contours)))
    crop_coords, segmented_img = get_crop_from_contours(image, contours)

    # reverse transformation
    segmented_tf = apply_inverse_transform(segmented_img)
    crop = apply_transform(crop_coords, coords=True)
    cropped_img = image_orig[crop[0][1]:crop[1][1], crop[0][0]:crop[1][0]]


    import matplotlib.pyplot as plt
    # Display the image and plot all contours found
    fig, ax = plt.subplots(ncols=3, figsize=(10, 3))
    ax[0].imshow(image_orig)
    ax[1].imshow(cropped_img)
    ax[2].imshow(segmented_tf)
    for a in ax:
        a.axis('image')
        a.set_xticks([])
        a.set_yticks([])
    plt.tight_layout()
    plt.savefig('test/crop_plot.png', dpi=500)
