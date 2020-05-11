from sklearn.model_selection import cross_val_score
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.metrics import accuracy_score

import sys
import numpy as np
import pandas as pd
sys.path.append('.')

from utils.file import get_filename_from_path, file_exists
from utils.timing import CodeTimer
from classifierutils import logger, default_params, standardize_data, load_dataset, classifier_dict


def eval_loo(X, Y, classifier):
    logger.debug("{} classifier".format(classifier))
    clf = classifier_dict[classifier]()
    if classifier in default_params:
        clf.set_params(**default_params[classifier])

    X = standardize_data(X, classifier)  # standardize data, method differs per classifier
    loo = LeaveOneOut()   # Leave-One-Out cross-validator
    cv_generator = loo.split(X)  # generates indices to split data into training & test set
    with CodeTimer() as timer:
        scores = cross_val_score(clf, X, Y, cv=cv_generator)  # list of accuracy score for each split
    time = timer.took
    accuracy = scores.mean()

    logger.info("-> Accuracy: {:0.2f}, Time: {}".format(accuracy, time))  # average score and inference time

    return accuracy, time, clf.get_params()


def default_evaluate():
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
                'histo_bins': int(histo_bins),
                'histo_eq': True if len(eq) else False,
                'accuracy': accuracy,
                'time': time
            })

    df = pd.DataFrame(rows)
    logger.info(df)
    df.to_csv('src/color_classifier/dataset_plots/eval_results.csv')


def hyperparams_grid_search(X, Y, params_grid, chosen_dataset):
    rows = []
    for classifier in params_grid.keys():
        logger.info(classifier)
        clf = classifier_dict[classifier]()

        X = standardize_data(X, classifier)  # standardize data, method differs per classifier

        parameters = params_grid[classifier]
        cv_generator = LeaveOneOut().split(X)
        clf = GridSearchCV(estimator=clf,  # classifier object eg. BernoulliNB()
                           param_grid=parameters,
                           scoring='accuracy',
                           cv=cv_generator,
                           verbose=5)
        clf.fit(X, Y)  # perform grid search to find best parameters
        best_params = clf.best_params_
        best_score = clf.best_score_
        print(clf.best_params_, clf.best_score_)

        # update classifier parameters
        clf_tuned = classifier_dict[classifier]().set_params(**best_params)
        if default_params.get(classifier) is not None:
            clf_default = classifier_dict[classifier]().set_params(**default_params.get(classifier))
        else:
            clf_default = classifier_dict[classifier]()

        # SAVE MEASUREMENTS
        # measure cross-validation computation time
        cv_generator = LeaveOneOut().split(X)
        with CodeTimer() as timer:
            scores = cross_val_score(clf_tuned, X, Y, cv=cv_generator)  # list of accuracy score for each split
        time_tuned = timer.took
        accuracy_tuned = scores.mean()

        # measure cross-validation computation time
        cv_generator = LeaveOneOut().split(X)
        with CodeTimer() as timer:
            scores = cross_val_score(clf_default, X, Y, cv=cv_generator)  # list of accuracy score for each split
        time_default = timer.took
        accuracy_default = scores.mean()

        rows.append({
            'classifier': classifier,
            'dataset': get_filename_from_path(chosen_dataset),
            'time_default': time_default,
            'accuracy_default': accuracy_default,
            'time_tuned': time_tuned,
            'accuracy_tuned': accuracy_tuned,
            'accuracy_best': best_score,
            'params_best': best_params
        })
    df = pd.DataFrame(rows)
    print(df)
    # df.to_csv('src/color_classifier/dataset_plots/{}-{}-tuning_results.csv'.format(
    #     get_filename_from_path(chosen_dataset, extension=False),
    #     '|'.join(list(params_grid.keys()))))
    df.set_index(['classifier', 'dataset'])
    if file_exists('src/color_classifier/dataset_plots/tuning_results.csv'):
        df.to_csv('src/color_classifier/dataset_plots/tuning_results.csv', mode='a', header=False)
    else:
        df.to_csv('src/color_classifier/dataset_plots/tuning_results.csv')


def tune_and_evaluate():
    params_grid = {
        #'MultinomialNB': {
        #     'alpha': np.geomspace(0.0001, 1.0, num=200, endpoint=True),
        #     'fit_prior': [True,False],
        #},
        'BernoulliNB': {
             'alpha': np.geomspace(0.0001, 1.0, num=50, endpoint=True),
             'fit_prior': [True,False],
             'binarize': np.linspace(0, 1.0, num=50, endpoint=False),
        },
        # 'SGDClassifier': {
        #     'loss': ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron'],
        #     'penalty': ['l2', 'l1', 'elasticnet'],
        #     'fit_intercept': [True, False],
            # 'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],

            # 'eta0': np.geomspace(0.25, 4, num=5),
        #     'max_iter': [4000]
        # },
        # 'Perceptron': {
        #     'alpha': np.geomspace(0.0001, 0.1, num=4, endpoint=True),
        #     'penalty': ['l2', 'l1', 'elasticnet', None],
        #     'fit_intercept': [True, False],
        #     'eta0': np.geomspace(0.25, 4, num=5),
        #     'max_iter': [4000]
        # },
        #'PassiveAggressiveClassifier': {
        #    'loss': ['hinge', 'squared_hinge'],
        #    'early_stopping': [True,False],
        #    'fit_intercept': [True, False],
        #    'C': np.geomspace(0.0001, 100.0, num=7, endpoint=True),
        #    'tol': np.geomspace(0.00001, 0.01, num=4, endpoint=True),
        #    'validation_fraction': [0.05, 0.1, 0.2],
        #    'n_iter_no_change': [1, 5, 10, 20],
        #    'tol': [0.0001, 0.001, 0.01],
        #    'max_iter': [1000,2000,4000],
        #    'average':[False,10,100]
        #},
        # 'MLPClassifier': {
        #     'alpha': np.linspace(0.0001, 1.0, num=5, endpoint=False),
        #     'solver': ['lbfgs'],
        #     'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #     'hidden_layer_sizes': [(10,), (50,), (100,)]
        # },
    }

    chosen_settings = [
        {
            'channels': 'hsv',
            'histo_bins': 32,
            'histo_eq': False,
        },
        {
            'channels': 'ycbcr',
            'histo_bins': 32,
            'histo_eq': False,
        },
    ]

    for chosen_setting in chosen_settings:
        chosen_dataset = 'src/color_classifier/dataset_json/dataset-{}-{}-{}.json'.format(
            chosen_setting['channels'],
            chosen_setting['histo_bins'],
            chosen_setting['histo_eq'] if chosen_setting['histo_eq'] else ''
        )

        X, Y = load_dataset(path=chosen_dataset)
        hyperparams_grid_search(X, Y, params_grid, chosen_dataset)


if __name__ == '__main__':
    tune_and_evaluate()
    # default_evaluate()
