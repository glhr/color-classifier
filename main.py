from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, gray2rgb
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import glob
import os.path

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import json

from utils.segmentation import get_segmentation_mask
from utils.contours import *

import matplotlib.pyplot as plt

X = []
Y = []

HISTO_BINS = 10


def get_rgb_histo(image, bins):
    image = rgb2hsv(image)
    if np.max(image) > 1:
        range = (0,255)
    else:
        range = (0,1)
    histo_r = np.histogram(image[:,:,0], range=range, bins=bins, density=True)
    histo_g = np.histogram(image[:,:,1], range=range, bins=bins, density=True)
    histo_b = np.histogram(image[:,:,2], range=range, bins=bins, density=True)

    vector = np.hstack((histo_r[0], histo_g[0], histo_b[0]))

    return vector


def get_color_from_filename(filename):
    color = filename.split(".")[0].split("\\")[-1].split("/")[-1].split("-")[0]
    return color


def normalize_img(image):
    # print(image.shape)
    if len(image.shape) < 3:
        image = gray2rgb(image)
    elif image.shape[-1] > 3:
        image = image[:,:,:3]
    # image = resize(image, (100,100,3))

    # print(image.shape)
    return image


def generate_dataset():
    for image_filename in glob.glob("dataset/*"):
        image = io.imread(image_filename)
        print(image_filename)


        image = normalize_img(image)

        if np.max(image) > 1:
            image = image/255

        try:
            image_value = image[:,:,2]
            contours = get_contours(image_value)
            masks = get_masks_from_contours(image_value, contours)
            masked = get_masked_image(image,masks)
            io.imsave('contours/'+image_filename.split("\\")[-1].split("/")[-1], masked)
        except:
            pass


def datasetToJSON():
    images = []
    for image_filename in glob.glob("contours/*"):
        print(image_filename)
        masked = io.imread(image_filename)

        histo = get_rgb_histo(masked, bins=HISTO_BINS)

        image_dict = {
            'filename': image_filename,
            # 'features':list(image_downscaled.flatten()),
            'histo': list(map(int, histo)),
            'color': get_color_from_filename(image_filename)
        }
        images.append(image_dict)

        # file = image_filename.split("\\")[-1].split("/")[-1]
        # image = io.imread('dataset/'+file)
        # image = normalize_img(image)
        # fig, ax = plt.subplots(ncols=2, figsize=(8, 3))
        # ax[0].imshow(image)
        # ax[1].imshow(masked)
        # for a in ax:
        #     a.axis('image')
        #     a.set_xticks([])
        #     a.set_yticks([])
        #
        # plt.tight_layout()
        # plt.savefig('plots_contours/{}.png'.format(file), dpi=300)

    with open('dataset.json', 'w') as json_file:
        json.dump(images, json_file)


# if it doesn't exist create JSON file from image dataset for training
# if not os.path.isfile('dataset.json'):
#     datasetToJSON()

# generate_dataset()
datasetToJSON()

# load image features and corresponding color classes
df = pd.read_json('dataset.json')
X = list(df['histo'])
Y = list(df['color'])
files = list(df['filename'])
print(df)


def get_model(X_train, y_train):
    # train
    # clf = SGDClassifier()
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


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


def test_img(image_filename):
    image_orig = io.imread(image_filename)
    image = normalize_img(image_orig)

    y_test = get_color_from_filename(image_filename)

    if np.max(image) > 1:
        image = image/255

    image_value = image[:,:,2]
    contours = get_contours(image_value)
    masks = get_masks_from_contours(image_value, contours)
    masked = get_masked_image(image,masks)
    io.imsave('test/masked-'+image_filename.split("\\")[-1].split("/")[-1], masked)
    histo = get_rgb_histo(masked, bins=HISTO_BINS)

    clf = get_model(X, Y)
    X_test = list(map(int, histo))
    predicted = clf.predict([X_test])
    print("Test:\t", y_test, '\n-->\t', predicted)

eval_split_dataset()

# test_img("test/green.png")
test_img("test/green-cropped.png")
test_img("test/green-cropped2.png")
