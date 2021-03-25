from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def random_forest_(x_tr, x_te, y_tr, y_te, n_estimators=100):
    """ Returns: (RF instance, score)"""
    rf = RandomForestClassifier(n_estimators=n_estimators, random_state=42)
    rf.fit(x_tr, y_tr)
    return rf, rf.score(x_te, y_te)


def svm_(x_tr, x_te, y_tr, y_te, cv=False, param_grid={}, c=1):
    """ Returns: (SVM instance, score)"""

    if cv: # cross validation
        print('\t\tTuning hyperparameters...')

        clf = GridSearchCV(SVC(), param_grid)
        clf.fit(x_tr, y_tr)

        print('\t\t... best value for C: ', clf.best_params_['C'])

        c_value = clf.best_params_['C']
    else:
        c_value = c

    svc = SVC(C=c_value, random_state=42)
    svc.fit(x_tr, y_tr)

    return svc, svc.score(x_te, y_te)


def knn_(x_tr, x_te, y_tr, y_te, cv=False, param_grid={}, neighbors=5):
    """ Returns: (KNN instance, score)"""

    if cv: # cross validation
        print('\t\tTuning hyperparameters...')

        clf = GridSearchCV(KNeighborsClassifier(), param_grid)
        clf.fit(x_tr, y_tr)

        n_value = clf.best_params_['n_neighbors']

        print('\t\t... best value for n_neighbors: ', n_value)
    else:
        n_value = neighbors

    knn = KNeighborsClassifier(n_neighbors=n_value)
    knn.fit(x_tr, y_tr)

    return knn, knn.score(x_te, y_te)
