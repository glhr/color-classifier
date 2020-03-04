from skimage import io
from skimage.transform import resize
from sklearn.model_selection import train_test_split
import numpy as np
import cv2
import glob

from sklearn import linear_model
import pandas as pd
import json

X = []
Y = []

def datasetToJSON():
    images = []
    for image_filename in glob.glob("dataset/*"):
        image = io.imread(image_filename)
        print(image_filename)
        color = image_filename.split(".")[0].split("\\")[-1].split("-")[0]
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
clf = linear_model.SGDClassifier()
clf.fit(X_train,y_train)

# test classifier on unknown image(s) from testing dataset
print("Test:\t", y_test, '\n-->\t', clf.predict(X_test))
