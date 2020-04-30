from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut

import sys
import pandas as pd
sys.path.append('.')

from utils.file import get_filename_from_path
from utils.timing import CodeTimer
from classifierutils import logger, classifier_params, standardize_data, load_dataset, classifier_dict


def eval_loo(X, Y, classifier):
    logger.debug("{} classifier".format(classifier))
    clf = classifier_dict[classifier]()
    if classifier in classifier_params:
        clf.set_params(**classifier_params[classifier])

    X = standardize_data(X, classifier)  # standardize data, method differs per classifier
    loo = LeaveOneOut()   # Leave-One-Out cross-validator
    cv_generator = loo.split(X)  # generates indices to split data into training & test set
    with CodeTimer() as timer:
        scores = cross_val_score(clf, X, Y, cv=cv_generator)  # list of accuracy score for each split
    time = timer.took
    accuracy = scores.mean()

    logger.info("-> Accuracy: {:0.2f}, Time: {}".format(accuracy, time))  # average score and inference time

    return accuracy, time, clf.get_params()


if __name__ == '__main__':
    # classifiers = [
    #     'MultinomialNB',
    #     'SGDClassifier',
    #     'Perceptron',
    #     'PassiveAggressiveClassifier',
    #     'BernoulliNB'
    # ]
    classifiers = list(classifier_dict.keys())
    # classifiers = ['SGDClassifier','BernoulliNB','MultinomialNB']
    rows = []
    import glob
    for dataset in glob.glob('src/color_classifier/dataset_json/*.json'):
        filename = get_filename_from_path(dataset, extension=False)
        logger.info(filename)
        channels, histo_bins, eq = filename.split('-')[1:]
        X, Y = load_dataset(path=dataset)
        # logger.info("Dataset size: {}".format(len(X)))
        for classifier in classifiers:
            accuracy, time, params = eval_loo(X, Y, classifier)
            rows.append({
                'classifier': classifier,
                'channels': channels,
                'histo_bins':int(histo_bins),
                'histo_eq': True if len(eq) else False,
                'accuracy': accuracy,
                'time': time
            })

    df = pd.DataFrame(rows)
    logger.info(df)
    df.to_csv('src/color_classifier/dataset_plots/eval_results.csv')
