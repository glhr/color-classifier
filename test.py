import numpy as np

from vision_utils.logger import get_logger
from vision_utils.img import normalize_img, image_to_numpy, numpy_to_image, bgr_to_rgb, adjust_image_range, save_image
from vision_utils.contours import get_object_crops, select_best_object
from vision_utils.plotting import plot_bounding_boxes
from vision_utils.timing import get_timestamp
from vision_utils.file import get_filename_from_path
from color_classifier.classifier import channels, classifier, classify_objects, add_training_image, update_model_with_user_data, initialize_classifier
from color_classifier.classifierutils import HISTO_BINS, PATH_TO_DATASET_JSON, best_params, classifier_dict
from skimage.io import imread

logger = get_logger()

salad_ref = {
    0: ['brown'],
    1: ['blue'],
    2: ['green'],
    3: [],
    4: ['red'],
    5: ['black'],
    6: ['yellow'],
    7: ['red'],
    8: ['brown'],
    9: ['red'],
    10: ['purple'],
    11: ['purple'],
    12: ['blue'],
    13: ['blue'],
    14: ['green'],
    15: ['orange'],
    16: ['yellow', 'green'],
    17: ['green'],
    18: ['red'],
    19: ['red'],
    20: ['brown'],
    21: ['black', 'blue', 'purple'],
    22: ['yellow'],
    23: ['orange'],
    24: ['red'],
    25: ['red'],
    26: ['green'],
    27: ['purple'],
    28: ['purple'],
    29: ['blue'],
    30: ['blue']
}


def process_image(image_orig, run_classifier=True, placement='any', learn=False, ref={}):
    global colors_detected
    global objects
    global IMG_SIZE
    colors_detected = []
    image = normalize_img(image_orig)  # make sure it's RGB & in range (0,1)
    IMG_SIZE = image[:,:,0].shape

    ## for testing blank image
    # image = np.zeros_like(image)
    # image_orig = np.zeros_like(image_orig)

    objects = get_object_crops(image, placement=placement, transform=True)
    logger.debug("Found {} objects".format(len(objects)))

    # if objects were found, run them through the classifier to get the color and plot the results
    if len(objects) > 0:
        if run_classifier and not learn:
            predictions = classify_objects(image, objects=objects)
        else:
            predictions = []

        image = adjust_image_range(image, max=255).astype(np.uint8)

        if learn:
            for object in objects:
                color = learn
                add_training_image(image, object, color)
                print("adding training image")
            img_boxes = plot_bounding_boxes(image, [object], [], None, show=False)

        else:
            colors_detected = predictions
            correct = 0
            incorrect = 0
            correct_indexes, ignore_indexes = [], []
            for i in range(len(objects)):
                # logger.debug("{}".format(objects[i].coords))
                if run_classifier:
                    # logger.debug("--> {}".format(predictions[i]))
                    if i in ref:
                        if predictions[i] in ref[i]:
                            correct += 1
                            correct_indexes.append(i)
                        elif not len(ref[i]):
                            ignore_indexes.append(i)
                            pass
                        else:
                            incorrect += 1
            print("{} correct, {} incorrect".format(correct, incorrect))

            img_boxes = plot_bounding_boxes(image, objects, predictions, None, show=False, numbering=False, correct_indexes=correct_indexes, ignore_indexes=ignore_indexes)

        return img_boxes, (correct, incorrect)
    else:
        logger.debug("No objects found :(")
        colors_detected = []
        image = adjust_image_range(image, max=255).astype(np.uint8)
        return image, None


dataset = get_filename_from_path(PATH_TO_DATASET_JSON, extension=True)


def save_classifier_output(test='', reference_color=None):
    # process and save a single image (for testing)
    logger.info("Saving and processing an image for testing")
    image = imread("test/{}.png".format(reference_color))
    img_boxes, scores = process_image(image, run_classifier=True, ref=salad_ref)
    if reference_color is not None:
        save_image(img_boxes, 'test/{}-{}-{}-{}-{}.png'.format(
            reference_color,
            classifier,
            HISTO_BINS,
            channels,
            test))
        print(colors_detected)
    else:
        save_image(img_boxes, 'test/{}-{}-{}-{}.png'.format(
            reference_color,
            classifier,
            HISTO_BINS,
            channels,
            test))
    masks = np.zeros_like(image[:,:,0])
    for i, object in enumerate(objects):
        mask = object.get_mask(type=np.uint8, range=255)
        # save_image(mask, 'test/{} test-mask{}.png'.format(timestamp, i))
        masks += mask
    return scores
    # save_image(masks, 'test/{}-mask.png'.format(reference_color))


# for color in ['red', 'green', 'black']:
    # save_classifier_output(reference_color=color)
    # image = imread("test/{}.png".format(color))
    # img_boxes = process_image(image, run_classifier=False, learn=color)


for color in ['salad']:
    results = []
    for channels in ['ycbcr']:
        for classifier in best_params[dataset].keys():
        # for classifier in ['MultinomialNB']:
            initialize_classifier(channels, classifier, use_best_params=False) # before tuning
            score_beforetuning = save_classifier_output(test='beforetuning', reference_color=color)
            initialize_classifier(channels, classifier, use_best_params=True) # after tuning
            score_beforelearning = save_classifier_output(test='beforelearning', reference_color=color)
            # initialize_classifier(channels, classifier, use_best_params=True) # after learning
            update_model_with_user_data()
            score_afterlearning = save_classifier_output(test='afterlearning', reference_color=color)

            summary = {
                'channels': channels,
                'classifier': classifier,
                'beforetuning': score_beforetuning,
                'beforelearning': score_beforelearning,
                'afterlearning': score_afterlearning
                }
            results.append(summary)

import pandas as pd
df = pd.DataFrame(results)
df.to_csv('test/learningresults.csv')
# update_model_with_user_data()
# save_classifier_output(afterlearning=True, reference_color=color)
