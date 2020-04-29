import pandas as pd
import json

from sklearn import preprocessing

from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from utils.logger import get_logger

PATH_TO_DATASET_JSON = 'src/color_classifier/dataset.json'
PATH_TO_DATASET_IMGS = 'src/color_classifier/dataset/'
PATH_TO_CONTOURS_IMGS = 'src/color_classifier/contours/'
HISTO_BINS = 10

logger = get_logger()


classifier_dict = {
    'MultinomialNB': MultinomialNB,
    'SGDClassifier': SGDClassifier,
    'Perceptron': Perceptron,
    'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
    'BernoulliNB': BernoulliNB
    # 'MLPClassifier': MLPClassifier
}


def get_model(X_train, y_train, classifier='MultinomialNB', standardize=False):
    """
    Given a list of feature vectors X, and a list of ground truth classes,
    train the linear classifier and return the model
    """
    if standardize and classifier=='SGDClassifier':
        X_train = preprocessing.scale(X_train)
    clf = classifier_dict[classifier]()
    clf.fit(X_train, y_train)
    return clf


def save_dataset(images, output_path=PATH_TO_DATASET_JSON):
    """
        images: list of dictionaries in the following format:
            image_dict = {
                'filename': image_filename,
                'histo': get_feature_vector(image, mask=mask, bins=histo_bins),
                'color': string
            }
    """
    with open(output_path, 'w') as json_file:
        json.dump(images, json_file)
        logger.info("Saved dataset to {}".format(output_path))


def load_dataset(path=PATH_TO_DATASET_JSON):
    # load image features and corresponding color classes
    df = pd.read_json(PATH_TO_DATASET_JSON)
    X = list(df['histo'])
    Y = list(df['color'])
    files = list(df['filename'])
    # print(df)
    return X, Y
