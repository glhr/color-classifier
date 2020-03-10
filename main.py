from skimage import io
from skimage.transform import resize
from skimage.color import rgb2hsv, gray2rgb
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import glob

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import json

X = []
Y = []

def datasetToJSON():
    images = []
    for image_filename in glob.glob("dataset/*"):
        image = io.imread(image_filename)
        print(image_filename)
        color = image_filename.split(".")[0].split("\\")[-1].split("/")[-1].split("-")[0]
        print(image.shape)
        if len(image.shape) < 3:
            image = gray2rgb(image)
        elif image.shape[-1] > 3:
            image = image[:,:,:3]
        # hsv_img = rgb2hsv(image)
        image_downscaled = resize(image, (10,10,3))

        image_dict = {
            'filename':image_filename,
            'features':list(image_downscaled.flatten()),
            'color': color
        }
        images.append(image_dict)

    with open('dataset.json', 'w') as json_file:
        json.dump(images , json_file)

    # cv2.imshow('img',image_downscaled)
    # cv2.waitKey(0)

import os.path

# if it doesn't exist create JSON file from image dataset for training
if not os.path.isfile('dataset.json'):
    datasetToJSON()

# load image features and corresponding color classes
df = pd.read_json('dataset.json')
df['features'] = df['features']
X = list(df['features'])
Y = list(df['color'])
files = list(df['filename'])
print(df)

# split dataset into training and testing
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.1)

# train
clf = MultinomialNB()
clf.fit(X_train,y_train)

# test classifier on unknown image(s) from testing dataset
predicted = clf.predict(X_test)
print("Test:\t", y_test, '\n-->\t', predicted)

# calculate ratio of correct predictions
correct = np.zeros(len(X_test))
for i in range(len(X_test)):
    correct[i] = 1 if (y_test[i] == predicted[i]) else 0
print("Ratio of correct predictions:", np.round(np.sum(correct)/len(X_test),2))
