from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

import sys
sys.path.append('.')

from classifierutils import logger, standardize_data, load_dataset, get_model, classifier_dict, PATH_TO_DATASET_JSON
from utils.file import file_exists


if not file_exists(PATH_TO_DATASET_JSON):
    PATH_TO_DATASET_JSON = PATH_TO_DATASET_JSON.split('/')[-1]  # try alternative path
    if not file_exists(PATH_TO_DATASET_JSON):
        raise FileNotFoundError("no dataset.json found, run generate_dataset() from classifier.py")

X_orig, Y = load_dataset()


def eval_loo(classifier):
    X = X_orig
    logger.debug("{} classifier".format(classifier))
    clf = get_model(X, Y, classifier=classifier, debug=True)

    X = standardize_data(X, classifier)

    loo = LeaveOneOut()   # Leave-One-Out cross-validator
    cv_generator = loo.split(X)  # generates indices to split data into training & test set
    scores = cross_val_score(clf, X, Y, cv=cv_generator)  # list of accuracy score for each split

    logger.info("-> Accuracy: {:0.2f}".format(scores.mean()))  # average score


if __name__ == '__main__':
    # classifiers = [
    #     'MultinomialNB',
    #     'SGDClassifier',
    #     'Perceptron',
    #     'PassiveAggressiveClassifier',
    #     'BernoulliNB'
    # ]
    # classifiers = list(classifier_dict.keys())
    classifiers = ['SGDClassifier','BernoulliNB','MultinomialNB']
    logger.info("Dataset size: {}".format(len(X_orig)))
    for classifier in classifiers:
        eval_loo(classifier)
