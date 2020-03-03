from skimage import io
from sklearn.feature_extraction import image
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import cv2
import glob

import numpy as np
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

# pick a random image from the dataset,
# remove it from the training dataset and use it for resting
import random
testing_n = random.randint(0,len(X)-1)
testing_image = X.pop(testing_n)
testing_class = Y.pop(testing_n)
testing_filename = files[testing_n]

# train
clf = linear_model.SGDClassifier()
clf.fit(X,Y)

# test classifier on unknown image
print("Test:", testing_filename, testing_class, '-->', clf.predict([testing_image]))
