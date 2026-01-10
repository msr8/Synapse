from flask_socketio import emit
from sklearn.model_selection import train_test_split

# https://chatgpt.com/share/680a67d5-d46c-8007-87cd-ac309bae15a2
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier

from skopt import BayesSearchCV
from skopt.callbacks import DeadlineStopper, ThresholdStopper, DeltaXStopper
from skopt.utils import OptimizeResult

from time import perf_counter
import pandas as pd
import numpy as np
from typing import Callable, Dict, Any, Tuple

from rich import print as printf, inspect

from warnings import simplefilter
simplefilter(action='ignore', category=RuntimeWarning) # "The objective has been evaluated at point [...] before, using random point []...]"
simplefilter(action='ignore', category=UserWarning)    # "invalid value encountered in cast"



def create_flask_callback(model:str) -> Callable[[OptimizeResult], None]:
    def flask_callback(result:OptimizeResult) -> None:
        emit('bayesian_callback', {
            'model': model,
            'n_iter': len(result.x_iters),
            'best_score': -result.fun, # Have to put a minus sign because the score is negative because of how skopt works
        })
    return flask_callback





def bayesian_search(x_train:pd.DataFrame, y_train:pd.Series, x_test:pd.DataFrame, y_test:pd.Series, model:str, gen_stopper:callable, scorer:str='accuracy', cv:int=3, n_iter:int=2) -> None:
    clf = MODELS[model]['model']()
    search_space = MODELS[model]['search_space']
    bayes_search_args = {
        'estimator': clf,
        'search_spaces': search_space | MODELS[model].get('fixed_params',{}),
        'cv': cv,
        'scoring': scorer,
        'n_jobs': -1,
    }
    if gen_stopper is None: bayes_search_args['n_iter'] = n_iter

    bayes_search = BayesSearchCV(**bayes_search_args)
    callbacks    = [create_flask_callback(model)]
    if gen_stopper is not None: callbacks.append(gen_stopper())

    time_start = perf_counter()
    bayes_search.fit(x_train, y_train, callback=callbacks)
    time_taken = perf_counter() - time_start

    testing_score = bayes_search.score(x_test, y_test)

    # inspect(bayes_search)
    return {
        'clf': bayes_search.best_estimator_,
        'best_training_score': bayes_search.best_score_,
        'testing_score': testing_score,
        'best_params': {k: v for k, v in bayes_search.best_params_.items() if k in search_space},
        'n_iter': len(bayes_search.cv_results_["params"]),
        'time_taken': time_taken
    }



def initialise_bayesian(df:pd.DataFrame, target:str, test_size_ratio:float=0.2, stratified:bool=True, stopper:str='n_iter', deadline_time:int=20, threshold_value:float=0.1, delta_value: float=0.01) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Any]:
    x = df.drop(columns=[target])
    y = df[target]

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size_ratio, stratify=y if stratified else None, random_state=42)

    match stopper:
        case 'n_iter':    gen_stopper = None
        case 'deadline':  gen_stopper = lambda: DeadlineStopper(deadline_time)
        case 'threshold': gen_stopper = lambda: ThresholdStopper(threshold_value)
        case 'delta':     gen_stopper = lambda: DeltaXStopper(delta_value)

    return x_train, x_test, y_train, y_test, gen_stopper
    



MODELS = {
    'knn': {
        'model': KNeighborsClassifier,
        'display_name': 'K Nearest Neighbors',
        'search_space': {
            'n_neighbors': (1, 30),
            'weights': ['uniform', 'distance'],
            'metric': ['chebyshev', 'cosine', 'euclidean', 'manhattan', 'minkowski', 'sqeuclidean']
        }
    },

    'SVM': {
        'model': SVC,
        'display_name': 'Support Vector Machine',
        'search_space': {
            'C': (1e-4, 1e+4, 'log-uniform'),
            'kernel': ['rbf','sigmoid', 'poly'], # 'linear' was having problems
            'degree': (1, 3),
            'gamma': ['scale'] # 'auto' was having problems
        }
    },

    'lr': {
        'model': LogisticRegression,
        'display_name': 'Logistic Regression',
        'search_space': {
            'C': (1e-6, 1e+6, 'log-uniform'),
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }
    },

    'dtree': {
        'model': DecisionTreeClassifier,
        'display_name': 'Decision Tree',
        'search_space': {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': (1, 10),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features': [None, 'sqrt', 'log2']
        }
    },

    'rf': {
        'model': RandomForestClassifier,
        'display_name': 'Random Forest',
        'search_space': {
            'n_estimators': (10, 100),
            'criterion': ['gini', 'entropy'],
            'max_depth': (1, 10),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features': [None, 'sqrt', 'log2']
        }
    },

    'extra-trees': {
        'model': ExtraTreesClassifier,
        'display_name': 'Extra Trees',
        'search_space': {
            'n_estimators': (10, 100),
            'criterion': ['gini', 'entropy'],
            'max_depth': (1, 10),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features': [None, 'sqrt', 'log2']
        }
    },

    'gradient_boosting': {
        'model': GradientBoostingClassifier,
        'display_name': 'Gradient Boosting',
        'search_space': {
            'n_estimators': (10, 100),
            'learning_rate': (1e-6, 1, 'log-uniform'),
            'max_depth': (1, 10),
            'min_samples_split': (2, 10),
            'min_samples_leaf': (1, 10),
            'max_features': [None, 'sqrt', 'log2']
        }
    },

    'lgbm': {
        'model': LGBMClassifier,
        'display_name': 'Light Gradient Boosting Machine',
        'search_space': {
            'n_estimators': (10, 100),
            'learning_rate': (1e-6, 1, 'log-uniform'),
            'max_depth': (-1, 15),
            'num_leaves': (10, 50),
            'min_child_samples': (5, 20),
            'subsample': (0.5, 1.0),
            'colsample_bytree': (0.5, 1.0),
            'reg_alpha': (0.0, 5.0),
            'reg_lambda': (0.0, 5.0)
        },
        'fixed_params': {
            'verbose': [-1],
        }
    },

    'adaboost': {
        'model': AdaBoostClassifier,
        'display_name': 'Ada Boost',
        'search_space': {
            'n_estimators': (10, 100),
            'learning_rate': (1e-6, 1, 'log-uniform'),
        }
    },

    'bagging': {
        'model': BaggingClassifier,
        'display_name': 'Bagging',
        'search_space': {
            'n_estimators': (10, 100),
            'max_samples': (0.1, 1.0),
            'max_features': (0.1, 1.0),
            'bootstrap': [True, False],
            'bootstrap_features': [True, False]
        }
    }
}


MODELS_DNS = { i: MODELS[i]['display_name'] for i in MODELS } 


