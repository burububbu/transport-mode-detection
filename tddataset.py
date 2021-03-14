import pandas as pd;
import utils;
import matplotlib.pyplot as plt;
import numpy as np

class TDDataset:
    """
    Transport detection dataset with sensor data.
    Fields:
        - feat
        - data
        - train_dt // preprocessed  by main
        - test_dt // preprocessed by main
    """
    def __init__(self, path, excluded_sensors=[]):
        # load data, after calc features to exclude
        data = pd.read_csv(path).iloc[:, 5:-1]
        
        def to_exclude(w):
            for f in excluded_sensors:
                if f in w:
                    return True
        
        self.data = data[[f for f in data.columns.values if not to_exclude(f)]]
        
        # add feat list (useful to retrieve then indexes)
        col_names = self.data.columns.values
        
        self.feat = [utils.clean_name(col_names[i]) for i in range(0, len(col_names)-1, 4)]
      
    def set_train_test(self, x_tr, y_tr, x_te, y_te):
        """Set train and test data"""
        self.train_dt = (x_tr, y_tr)
        self.test_dt = (x_te, y_te)    
    
    def get_train_test_feat(self, sensors = []):
        """Get train data and test data with specific columns
           Return: tuple (train_x,  test_x, train_y, test_y)
        """
        indexes = []
        
        if sensors:
            for s in sensors:
                ind = self._get_i(s)
                indexes.extend(list(range(ind, ind + 4)))
        
        return (self.train_dt[0].iloc[:, indexes], self.test_dt[0].iloc[:, indexes],
                self.train_dt[1], self.test_dt[1] )
    
    def _get_i(self, name):
        ''' Get index of first sensor feature'''
        return self.feat.index(name) * 4
    
    def remove_sensor_feat(self, name):
        if name in self.feat:
            ind = self._get_i(name)
            self.data.drop(self.data.columns[list(range(ind, ind+4))], axis=1, inplace=True)
            self.feat.remove(name)
    
    
    