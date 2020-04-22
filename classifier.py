from skimage import io
from sklearn.model_selection import train_test_split
import numpy as np
import glob

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import json

import sys
sys.path.append('.')
from utils.segmentation import get_segmentation_mask
from utils.contours import is_mask_an_object, get_contours, get_masks_from_contours, get_masked_image, select_best_mask, get_mask_area
from utils.img import normalize_img, get_hsv_histo, get_2d_image
from utils.file import get_color_from_filename, get_working_directory, get_filename_from_path, file_exists
from utils.logger import get_logger

import matplotlib.pyplot as plt

X = []
Y = []

HISTO_BINS = 10

PATH_TO_DATASET_JSON = 'src/color_classifier/dataset.json'
PATH_TO_DATASET_IMGS = 'src/color_classifier/dataset/'
PATH_TO_CONTOURS_IMGS = 'src/color_classifier/contours/'

logger = get_logger()


def generate_dataset():
    images = []
    for image_filename in glob.glob(PATH_TO_DATASET_IMGS+"*"):
        image = io.imread(image_filename)
        logger.debug(image_filename)

        image = normalize_img(image)

        try:
            image_value = get_2d_image(image)
            contours = get_contours(image_value)
            masks = get_masks_from_contours(image_value, contours)
            mask = select_best_mask(image, masks, filter=False)
            masked, mask = get_masked_image(image, mask)
            logger.debug(PATH_TO_CONTOURS_IMGS+get_filename_from_path(image_filename))
            io.imsave(PATH_TO_CONTOURS_IMGS+get_filename_from_path(image_filename), masked)

            histo = get_hsv_histo(masked, mask=mask, bins=HISTO_BINS)

            image_dict = {
                'filename': image_filename,
                # 'features':list(image_downscaled.flatten()),
                'histo': list(map(int, histo)),
                'color': get_color_from_filename(image_filename)
            }
            images.append(image_dict)
        except Exception as e:
            logger.exception(e)
            pass
    with open(PATH_TO_DATASET_JSON, 'w') as json_file:
        json.dump(images, json_file)
        logger.info("Saved dataset to {}".format(PATH_TO_DATASET_JSON))


def save_contours():
    for image_filename in glob.glob(PATH_TO_CONTOURS_IMGS+"/*"):
        print(image_filename)
        masked = io.imread(image_filename)

        image = io.imread('dataset/'+get_filename_from_path(image_filename))
        image = normalize_img(image)
        fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
        ax[0].imshow(image)
        ax[1].imshow(masked)
        for a in ax:
            a.axis('image')
            a.set_xticks([])
            a.set_yticks([])

        plt.tight_layout()
        plt.savefig('plots_contours/{}.png'.format(get_filename_from_path(image_filename, extension=False)), dpi=300)


# if it doesn't exist create JSON file from image dataset for training
if not file_exists(PATH_TO_DATASET_JSON):
    generate_dataset()
# generate_dataset()

# load image features and corresponding color classes
df = pd.read_json(PATH_TO_DATASET_JSON)
X = list(df['histo'])
Y = list(df['color'])
files = list(df['filename'])
# print(df)


def get_model(X_train=X, y_train=Y):
    # train
    # clf = SGDClassifier()
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


clf = get_model(X, Y)


def eval_split_dataset():

    # split dataset into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.02)

    clf = get_model(X_train, y_train)

    # test classifier on unknown image(s) from testing dataset
    predicted = clf.predict(X_test)
    print("Test:\t", y_test, '\n-->\t', predicted)

    # calculate ratio of correct predictions
    correct = np.zeros(len(X_test))
    for i in range(len(X_test)):
        correct[i] = 1 if (y_test[i] == predicted[i]) else 0
    print("Ratio of correct predictions:", np.round(np.sum(correct)/len(X_test),2))


def classify_img(image, masks=None, select_mask='all', save=False, filepath=None):
    global clf
    image_value = get_2d_image(image)

    if masks is None:
        contours = get_contours(image_value)
        masks = get_masks_from_contours(image_value, contours)

    if select_mask == 'best':
        mask = select_best_mask(image, masks)
        masked, mask = get_masked_image(image, mask)
        histo = get_hsv_histo(masked, mask=mask, bins=HISTO_BINS)
        clf = get_model()
        X_test = list(map(int, histo))
        return clf.predict([X_test])
    elif select_mask == 'all':
        X_test = []
        for i, mask in enumerate(masks):
            # if is_mask_an_object(mask):
            masked, mask = get_masked_image(image, mask)

            histo = get_hsv_histo(masked, mask=mask, bins=HISTO_BINS)
            X_test.append(list(map(int, histo)))

            if save and filepath is not None:
                io.imsave(get_working_directory()+'/test/masked-'+str(i)+get_filename_from_path(filepath), masked)

        return clf.predict(X_test)


def test_img(image_filename):
    image_orig = io.imread(image_filename)
    image = normalize_img(image_orig)

    predicted = classify_img(image, save=False, filepath=image_filename)

    y_test = get_color_from_filename(image_filename)
    print("Test:\t", y_test, '\n-->\t', predicted)


if __name__ == '__main__':
    eval_split_dataset()

    # test_img("test/green.png")
    test_img(get_working_directory()+"/test/masked-1ros.png")
    test_img(get_working_directory()+"/test/green-cropped.png")
    test_img(get_working_directory()+"/test/green-cropped2.png")
