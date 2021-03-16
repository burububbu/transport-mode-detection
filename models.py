from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV

from sklearn.svm import SVC


def random_forest_(x_tr, x_te, y_tr, y_te, est = 100):
    rf = RandomForestClassifier(n_estimators = est, random_state= 42)
    rf.fit(x_tr, y_tr)
    return rf, rf.score(x_te, y_te)

def svm_(x_tr, x_te, y_tr, y_te, cv=False, param_grid = {}):
    # cross validation ( search for hyperparameter C )
    svc = SVC()
    if cv:
        # svc.score(x_te, y_te)
        clf = GridSearchCV(svc, param_grid)
        clf.fit(x_tr, y_tr)
        svc = SVC(C=clf.best_params_['C'])
    
    svc.fit(x_tr, y_tr)
    
    return svc, svc.score(x_te, y_te)

