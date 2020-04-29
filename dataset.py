import matplotlib.pyplot as plt
import glob

from utils.file import get_color_from_filename
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
                     equalize_histo=False,
                     img_path=PATH_TO_DATASET_IMGS,
                     output_path=PATH_TO_DATASET_JSON):

    images = []
    n_images = 0
    mosaics = dict()

    for color in ['green','blue','red','orange','yellow','black']:
    # for color in ['yellow']:
        thumbnails = []
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
                    thumbnails.append(masked)

                    # logger.debug(PATH_TO_CONTOURS_IMGS+get_filename_from_path(image_filename))
                    # io.imsave(PATH_TO_CONTOURS_IMGS+get_filename_from_path(image_filename), masked)

                    image_dict = {
                        'filename': image_filename,
                        # 'features':list(image_downscaled.flatten()),
                        'histo': get_feature_vector(image, mask=mask, bins=histo_bins),
                        'color': get_color_from_filename(image_filename)
                    }
                    images.append(image_dict)
            except Exception as e:
                logger.exception(e)
                pass
        mosaics[color] = create_mosaic(images=thumbnails, rows_first=False)
        save_image(mosaics[color], PATH_TO_CONTOURS_IMGS+color+'.png')
        # plt.figure(1)
        # plt.imshow(mosaics[color])
        # plt.show()
        # save_dataset(images)


if __name__ == '__main__':

    settings = {
        'mask_method': 'polygon',
        'resize_shape': (100, 100),
        'histo_bins': 10,
        'equalize_histo': False,
    }

    generate_dataset(mask_method=settings['mask_method'],
                     resize_shape=settings['resize_shape'],
                     histo_bins=settings['histo_bins'],
                     equalize_histo=settings['equalize_histo'],
                     img_path=PATH_TO_DATASET_IMGS,
                     output_path=PATH_TO_DATASET_JSON)
