import matplotlib.pyplot as plt
import glob

from utils.file import get_color_from_filename, file_exists
from utils.contours import get_contours, Object, select_best_object
from utils.img import save_image, normalize_img, get_feature_vector, get_2d_image, create_mosaic, load_image

try:
    from .classifierutils import save_dataset, logger, PATH_TO_CONTOURS_IMGS, PATH_TO_DATASET_JSON, PATH_TO_DATASET_IMGS
except ImportError:
    import sys
    sys.path.append('.')
    from classifierutils import save_dataset, logger, PATH_TO_CONTOURS_IMGS, PATH_TO_DATASET_JSON, PATH_TO_DATASET_IMGS

import numpy as np

def generate_dataset(mask_method='polygon',
                     resize_shape=(100, 100),
                     histo_bins=10,
                     histo_channels='hsv',
                     equalize_histo=False,
                     img_path=PATH_TO_DATASET_IMGS,
                     output_path=PATH_TO_DATASET_JSON.split('-')[0]+'.json',
                     overwrite_existing=False):

    output_path = "{}-{}-{}-{}.json".format(output_path.split('.')[-2], histo_channels, histo_bins, 'eq' if equalize_histo else '')

    if not overwrite_existing and file_exists(output_path):
        logger.info("{} already exists, no new dataset generated".format(output_path))
        return

    images = []
    n_images = 0
    mosaics_masked = dict()
    mosaics_orig = dict()

    for color in ['green','blue','red','orange','yellow','black','brown','purple']:
    # for color in ['purple']:
        thumbnails_masked = []
        thumbnails_orig = []
        for image_filename in glob.glob(img_path+color+"*"):
            n_images += 1
            # if n_images > 8:
            #     break
            image = load_image(image_filename)
            logger.debug(image_filename)

            try:
                image = normalize_img(image, resize_shape=resize_shape)
                image_value = get_2d_image(image, equalize_histo=equalize_histo)

                best_objects = []
                for level in np.linspace(0.05, 1, 19, endpoint=False):
                    contours = get_contours(image_value, level=level)
                    objects = [Object(contour, image, method=mask_method) for contour in contours]
                    if len(objects):
                        object = select_best_object(objects, constraints=None)
                        best_objects.append(object)

                if len(best_objects):
                    object = select_best_object(best_objects, constraints=None)
                    mask = object.get_mask(type=bool)

                    masked = object.get_masked_image()
                    thumbnails_masked.append(masked)
                    thumbnails_orig.append(image)

                    # logger.debug(PATH_TO_CONTOURS_IMGS+get_filename_from_path(image_filename))
                    # io.imsave(PATH_TO_CONTOURS_IMGS+get_filename_from_path(image_filename), masked)

                    image_dict = {
                        'filename': image_filename,
                        # 'features':list(image_downscaled.flatten()),
                        'histo': get_feature_vector(image,
                                                    mask=mask,
                                                    bins=histo_bins,
                                                    channels=histo_channels),
                        'color': get_color_from_filename(image_filename)
                    }
                    images.append(image_dict)
            except Exception as e:
                logger.exception(e)
                pass
        # mosaics_masked[color] = create_mosaic(images=thumbnails_masked, rows_first=False)
        # mosaics_orig[color] = create_mosaic(images=thumbnails_orig, rows_first=False)
        # save_image(mosaics_masked[color], PATH_TO_CONTOURS_IMGS+color+'.png')
        # save_image(mosaics_orig[color], PATH_TO_CONTOURS_IMGS+color+'_orig.png')
        # plt.figure(1)
        # plt.imshow(mosaics[color])
        # plt.show()

    save_dataset(images, output_path)


if __name__ == '__main__':

    histo_bins = [1, 2, 4, 8, 16, 32, 64, 128, 256]
    histo_channels = ['ycbcr', 'rgb', 'hsv']
    settings_list = [
        {
            'histo_bins': bins,
            'histo_channels': channels
        } for bins in histo_bins for channels in histo_channels
    ]
    print(settings_list)

    for settings in settings_list:
        generate_dataset(img_path=PATH_TO_DATASET_IMGS,
                         output_path=PATH_TO_DATASET_JSON.split('-')[0]+'.json',
                         **settings,)
