from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

import sys
sys.path.append('.')

from classifierutils import logger, classifier_params, standardize_data, load_dataset, classifier_dict


def eval_loo(X, Y, classifier):
    logger.debug("{} classifier".format(classifier))
    clf = classifier_dict[classifier]()
    if classifier in classifier_params:
        clf.set_params(**classifier_params[classifier])
        logger.debug(clf.get_params())

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
    import glob
    for dataset in glob.glob('src/color_classifier/dataset_json/*.json'):
        logger.info(dataset)
        X, Y = load_dataset(path=dataset)
        logger.info("Dataset size: {}".format(len(X)))
        for classifier in classifiers:
            eval_loo(X, Y, classifier)
