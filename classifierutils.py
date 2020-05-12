import pandas as pd
import json
import numpy as np

from sklearn import preprocessing

from sklearn.linear_model import SGDClassifier, Perceptron, PassiveAggressiveClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from utils.logger import get_logger

CLASSES = ['green', 'yellow', 'brown', 'black', 'blue', 'red', 'orange', 'purple']
HISTO_BINS = 32
CHANNELS = 'hsv'

PATH_TO_DATASET_JSON = 'src/color_classifier/dataset_json/dataset-{}-{}-.json'.format(CHANNELS, HISTO_BINS)
PATH_TO_DATASET_IMGS = 'src/color_classifier/dataset_img/'
PATH_TO_CONTOURS_IMGS = 'src/color_classifier/dataset_plots/'

logger = get_logger()

default_params = {
    'MLPClassifier': {
        'solver': 'lbfgs'
    },
    'MultinomialNB': {
        # 'alpha': 0.5,
        'fit_prior': False
    },
    'BernoulliNB': {
        # 'binarize': 0.5,
        'fit_prior': False
    }
}

best_params = {
    'dataset-hsv-32-.json': {
        'MultinomialNB': {
            'alpha': 0.002221946860939524,
            'fit_prior': False
        },
        'BernoulliNB': {
            'alpha': 0.0001,
            'binarize': 0.32,
            'fit_prior': True
        }
    },
    'dataset-ycbcr-32-.json': {
        'MultinomialNB': {
            'alpha': 0.0020255019392306666,
            'fit_prior': False
        },
        'BernoulliNB': {
            'alpha': 0.0024420530945486497,
            'binarize': 0.2,
            'fit_prior': False
        }
    }
}

#
# classifier_dict = {
#     'linear': {
#         'SGDClassifier': SGDClassifier,
#         'Perceptron': Perceptron,
#         'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
#         },
#     'proba': {
#         'MultinomialNB': MultinomialNB,
#         'BernoulliNB': BernoulliNB
#     },
#     'neural': {
#         'MLPClassifier': MLPClassifier
#     }
# }

classifier_dict = {
        'SGDClassifier': SGDClassifier,
        'Perceptron': Perceptron,
        'PassiveAggressiveClassifier': PassiveAggressiveClassifier,
        'MultinomialNB': MultinomialNB,
        'BernoulliNB': BernoulliNB,
        'MLPClassifier': MLPClassifier
}


def standardize_data(X_train, classifier):

    if classifier in ['SGDClassifier', 'Perceptron', 'PassiveAggressiveClassifier', 'MLPClassifier']:
        X_train = preprocessing.scale(X_train)
    elif classifier in ['BernoulliNB', 'MultinomialNB']:
        X_train = preprocessing.minmax_scale(X_train)
    # print(np.min(X_train), np.max(X_train))
    return X_train


def get_model(X_train,
              y_train,
              classifier='MultinomialNB',
              dataset='dataset-{}-{}-.json'.format(CHANNELS, HISTO_BINS),
              partial=False,
              debug=False):
    """
    Given a list of feature vectors X, and a list of ground truth classes,
    train the linear classifier and return the model
    """
    clf = classifier_dict[classifier]()
    if dataset in best_params:
        if classifier in best_params[dataset]:
            clf.set_params(**best_params[dataset][classifier])
            logger.info("Loading tuned parameters for {}".format(classifier))
    elif classifier in default_params:
        logger.info("Loading default parameters for {}".format(classifier))
        clf.set_params(**default_params[classifier])

    X_train = standardize_data(X_train, classifier)
    if not partial:
        clf.fit(X_train, y_train)
    else:
        clf.partial_fit(X_train, y_train, classes=CLASSES)
    if debug:
        logger.debug(clf.get_params())
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
    df = pd.read_json(path)
    X = list(df['histo'])
    Y = list(df['color'])
    files = list(df['filename'])
    # print(np.min(X), np.max(X))
    # print(df)
    return X, Y
