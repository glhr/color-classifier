from skimage import io

from utils.img import normalize_img, get_2d_image, get_feature_vector
from utils.contours import get_contours, Object
from utils.file import get_color_from_filename, get_working_directory, get_filename_from_path, file_exists
from .classifierutils import logger, load_dataset, best_params, get_model, PATH_TO_DATASET_JSON, HISTO_BINS, CHANNELS
from .dataset import generate_dataset

X = []
Y = []

# if it doesn't exist create JSON file from image dataset for training
if not file_exists(PATH_TO_DATASET_JSON):
    print("Generating dataset"+PATH_TO_DATASET_JSON)
    generate_dataset()
# generate_dataset(mask_method='polygon')
# generate_dataset(mask_method='binary_fill')

classifier = 'MultinomialNB'
X, Y = load_dataset()
clf = get_model(X, Y, classifier=classifier)

dataset = get_filename_from_path(PATH_TO_DATASET_JSON, extension=True)
if dataset in best_params:
    if classifier in best_params[dataset]:
        clf.set_params(**best_params[dataset][classifier])
        logger.info("Loading tuned parameters for {}".format(classifier))


def classify_objects(image, objects=None, save=False, filepath=None):
    """
    Given a list of n objects, return a list of n corresponding classes.
    If no object list is provided, the objects are first generated from contours in the image
    The classification is done using the globally defined model (clf)
    """
    global clf
    image_value = get_2d_image(image)

    logger.debug("Classifier {} params: {}".format(classifier,clf.get_params()))

    if objects is None:
        contours = get_contours(image_value)
        objects = [Object(contour, image) for contour in contours]

    X_test = []
    for i, object in enumerate(objects):
        mask = object.get_mask(type=bool)
        X_test.append(get_feature_vector(image, mask=mask, bins=HISTO_BINS, channels=CHANNELS))

        if save and filepath is not None:
            masked = object.get_masked_image()
            io.imsave(get_working_directory()+'/test/masked-'+str(i)+get_filename_from_path(filepath), masked)

    return clf.predict(X_test)


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
