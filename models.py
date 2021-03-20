from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def random_forest_(x_tr, x_te, y_tr, y_te, settings = {'n_estimators': 100}):
    rf = RandomForestClassifier(n_estimators=settings['n_estimators'], random_state=42)
    rf.fit(x_tr, y_tr)
    return rf, rf.score(x_te, y_te)

def svm_(x_tr, x_te, y_tr, y_te, settings={'cv': False, 'param_grid' : {}}):
    # cross validation ( search for hyperparameter C )
    svc = SVC()
    if settings['cv']:
        # svc.score(x_te, y_te)
        print('\t\tTuning hyperparameters...')
        clf = GridSearchCV(svc, settings['param_grid'])
        clf.fit(x_tr, y_tr)
        print('\t\t... best value for C: ', clf.best_params_['C'])
        svc = SVC(C=clf.best_params_['C'], random_state=42)
     
    svc.fit(x_tr, y_tr)
    
    return svc, svc.score(x_te, y_te)

def decision_tree_(x_tr, x_te, y_tr, y_te, settings= {'depth' : 4}):
    trc = DecisionTreeClassifier(max_depth=settings['depth'], random_state=42)
    trc.fit(x_tr, y_tr)
    return trc, trc.score(x_te, y_te)

def knn_(x_tr, x_te, y_tr, y_te, settings= {'cv': False, 'param_grid' : {}}):
    knn = KNeighborsClassifier()
    if settings['cv']:
        print('\t\tTuning hyperparameters...')
        clf = GridSearchCV(knn, settings['param_grid'])
        clf.fit(x_tr, y_tr)
        print('\t\t... best value for n_neighbors: ', clf.best_params_['n_neighbors'])
        knn = KNeighborsClassifier(n_neighbors=clf.best_params_['n_neighbors'])
    
    knn.fit(x_tr, y_tr)
    # , clf.cv_results_
    return knn, knn.score(x_te, y_te)


