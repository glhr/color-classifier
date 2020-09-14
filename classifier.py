from skimage import io

from vision_utils.img import normalize_img, get_2d_image, get_feature_vector, save_image
from vision_utils.contours import get_contours, Object
from vision_utils.file import get_color_from_filename, get_working_directory, get_filename_from_path, file_exists
from .classifierutils import get_dataset_name, logger, load_dataset, best_params, get_model, dataset_folder, HISTO_BINS, CHANNELS, CLASSIFIER
from .dataset import generate_dataset
from vision_utils.timing import CodeTimer, get_timestamp

import numpy as np

import json

X = []
Y = []

classifier = CLASSIFIER
channels = CHANNELS

dataset = dataset_folder + '/dataset_json/' + get_dataset_name(channels=channels)
dataset_user = dataset_folder + '/dataset_user/'
dataset_user_path = '{}dataset-{}-{}-.json'.format(dataset_user, channels, HISTO_BINS)

print(dataset)
# if it doesn't exist create JSON file from image dataset for training
if not file_exists(dataset):
    logger.error("Dataset file {} not found".format(dataset))
    raise FileNotFoundError
    print("Generating dataset"+dataset)
    generate_dataset()
# generate_dataset(mask_method='polygon')
# generate_dataset(mask_method='binary_fill')

global clf
CLASSIFIER_INITIALIZED = False

def initialize_classifier(channels_updated=channels, classifier_updated=classifier, use_best_params=True):
    global classifier, channels, clf, X, Y, dataset, dataset_user_path, CLASSIFIER_INITIALIZED
    dataset = dataset_folder + '/dataset_json/' + get_dataset_name(channels=channels_updated)
    dataset_user_path = '{}dataset-{}-{}-.json'.format(dataset_user, channels, HISTO_BINS)
    X, Y = load_dataset(path=dataset)
    classifier = classifier_updated
    logger.info(classifier)
    channels = channels_updated
    with CodeTimer() as timer:
        clf = get_model(X, Y, classifier=classifier, partial=True, use_best_params=use_best_params)
    logger.debug("{} took {} to train".format(classifier, timer.took))
    try:
        logger.debug("Initial partial fit, number of samples seen by model {}".format(
            list(zip(clf.classes_, clf.class_count_))))
    except AttributeError:
        pass
    CLASSIFIER_INITIALIZED = True



# dataset = get_filename_from_path(dataset, extension=True)
# if dataset in best_params:
#     if classifier in best_params[dataset]:
#         clf.set_params(**best_params[dataset][classifier])
#         logger.info("Loading tuned parameters for {}".format(classifier))


def classify_objects(image, objects=None, save=False, filepath=None):
    """
    Given a list of n objects, return a list of n corresponding classes.
    If no object list is provided, the objects are first generated from contours in the image
    The classification is done using the globally defined model (clf)
    """
    global clf
    if not CLASSIFIER_INITIALIZED:
        initialize_classifier()
    image_value = get_2d_image(image)

    logger.debug("Classifier {} params: {}".format(classifier,clf.get_params()))

    if objects is None:
        contours = get_contours(image_value)
        objects = [Object(contour, image) for contour in contours]

    X_test = []
    for i, object in enumerate(objects):
        mask = object.get_mask(type=bool)
        X_test.append(get_feature_vector(image, mask=mask, bins=HISTO_BINS, channels=channels))

        if save and filepath is not None:
            masked = object.get_masked_image()
            io.imsave(get_working_directory()+'/test/masked-'+str(i)+get_filename_from_path(filepath), masked)

    return clf.predict(X_test)

def add_training_image(image, object, color):
    global clf

    if not CLASSIFIER_INITIALIZED:
        initialize_classifier()

    logger.warning("Classifier: adding training image with color {} for {}".format(color, channels))

    mask = object.get_mask(type=bool)
    cropped_img = object.get_crop()
    path = '{}{}-{}.png'.format(dataset_user, color, get_timestamp())
    save_image(cropped_img, path)
    X = get_feature_vector(image,
                           mask=mask,
                           bins=HISTO_BINS,
                           channels=channels)
    image_dict = {
        'filename': path,
        'histo': X,
        'color': color
    }

    if file_exists(dataset_user_path):
        with open(dataset_user_path, 'r') as f:
            data = json.load(f)
            data.append(image_dict)
        with open(dataset_user_path, 'w') as f:
            json.dump(data, f)
    else:
        with open(dataset_user_path, 'w') as f:
            data = [image_dict]
            json.dump(data, f)

    # print(image_dict)
    clf.partial_fit(X=[X], y=[color], sample_weight=[2])
    logger.debug("After partial fit on {}, number of samples seen by model {}".format(
        [color],
        list(zip(clf.classes_, clf.class_count_))))
    return None


def update_model_with_user_data():
    logger.warning(dataset_user_path)
    if file_exists(dataset_user_path):
        with open(dataset_user_path, 'r') as f:
            data = json.load(f)

        for entry in data:
            X = entry['histo']
            if abs(np.sum(X)) > 0:
                y = entry['color']
                try:
                    clf.partial_fit(X=[X], y=[y], sample_weight=[2])
                except Exception:
                    clf.partial_fit(X=[X], y=[y])
                try:
                    logger.debug("After partial fit on {}, number of samples seen by model {}".format(
                        [y],
                        list(zip(clf.classes_, clf.class_count_))))
                except AttributeError:
                    pass


if __name__ == '__main__':

    def test_img(image_filename):
        image_orig = io.imread(image_filename)
        image = normalize_img(image_orig)

        predicted = classify_objects(image, save=False, filepath=image_filename)

        y_test = get_color_from_filename(image_filename)
        print("Test:\t", y_test, '\n-->\t', predicted)

    # test_img("test/green.png")
    test_img(get_working_directory()+"/test/masked-1ros.png")
    test_img(get_working_directory()+"/test/green-cropped.png")
    test_img(get_working_directory()+"/test/green-cropped2.png")
