from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import LeaveOneOut
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
import pandas as pd

import sys
sys.path.append('.')
from utils.file import file_exists
from utils.logger import get_logger

X = []
Y = []

PATH_TO_DATASET_JSON = 'src/color_classifier/dataset.json'

logger = get_logger()

if not file_exists(PATH_TO_DATASET_JSON):
    PATH_TO_DATASET_JSON = PATH_TO_DATASET_JSON.split('/')[-1]  # try alternative path
    if not file_exists(PATH_TO_DATASET_JSON):
        raise FileNotFoundError("no dataset.json found, run generate_dataset() from classifier.py")
# generate_dataset(mask_method='binary_fill')

# load image features and corresponding color classes
df = pd.read_json(PATH_TO_DATASET_JSON)
X = list(df['histo'])
Y = list(df['color'])
files = list(df['filename'])
# print(df)


def get_model(X_train=X, y_train=Y, classifier='MultinomialNB'):
    """
    Given a list of feature vectors X, and a list of ground truth classes,
    train the linear classifier and return the model
    """
    if classifier == 'SGDClassifier':
        clf = SGDClassifier()
    elif classifier == 'MultinomialNB':
        clf = MultinomialNB()
    clf.fit(X_train, y_train)
    return clf


def eval_loo(classifier):
    print("Performing LOO CV using {} classifier".format(classifier))
    clf = get_model(X, Y, classifier=classifier)
    loo = LeaveOneOut()   # Leave-One-Out cross-validator
    cv_generator = loo.split(X)  # generates indices to split data into training & test set
    scores = cross_val_score(clf, X, Y, cv=cv_generator)  # list of accuracy score for each split

    print("-> {} scores generated from dataset of size {}".format(len(scores), len(X)))
    print("-> Accuracy: {:0.2f}".format(scores.mean()))  # average score


if __name__ == '__main__':
    classifiers = ['MultinomialNB', 'SGDClassifier']
    for classifier in classifiers:
        eval_loo(classifier)
