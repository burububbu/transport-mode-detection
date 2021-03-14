from sklearn.ensemble import RandomForestClassifier
import pandas as pd

def random_forest(x_tr, x_te, y_tr, y_te, est = 100):
    ''' Return accuracy + rank val series'''
    rf = RandomForestClassifier(n_estimators = est, random_state= 42)
    rf.fit(x_tr, y_tr)
    rank_val = pd.Series(rf.feature_importances_, index=x_tr.columns)
    return {'accuracy': rf.score(x_te, y_te), 'rank': rank_val }
