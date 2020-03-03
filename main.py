from skimage import io
from sklearn.feature_extraction import image
from skimage.transform import rescale, resize, downscale_local_mean
import numpy as np
import cv2
import glob

import numpy as np
from sklearn import linear_model

X = []
Y = []

for image_filename in glob.glob("dataset/*"):
    image = io.imread(image_filename)
    print(image_filename)
    color = image_filename.split(".")[0].split("\\")[-1].split("-")[0]
    image_downscaled = resize(image, (10,10,3))

    X.append(image_downscaled.flatten())
    Y.append(color)

    # cv2.imshow('img',image_downscaled)
    # cv2.waitKey(0)

# pick a random image from the dataset,
# remove it from the training dataset and use it for resting
import random
testing_n = random.randint(0,len(X)-1)
testing_image = X.pop(testing_n)
testing_class = Y.pop(testing_n)
print(testing_class)

clf = linear_model.SGDClassifier()
clf.fit(X,Y)

print(clf.predict([testing_image]))
