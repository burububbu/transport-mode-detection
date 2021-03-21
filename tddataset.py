import pandas as pd;
import utils;
import matplotlib.pyplot as plt;
import numpy as np
import preprocessing as pre
from sklearn.model_selection import train_test_split

class TDDataset:
    """
    Transport detection dataset with sensor data.
    Fields:
        - feat = ['accellerometer', ...]
        - data
        - train_dt = (x_tr, y_tr) 
        - test_dt = (x_te, x_te) 
    """
    def __init__(self, path, excluded_sensors=[]):

        data = pd.read_csv(path).iloc[:, 5:-1]
        
        def to_exclude(w):
            for f in excluded_sensors:
                if f in w:
                    return True
        
        self.data = data[[f for f in data.columns.values if not to_exclude(f)]]
        
        # add feat list (useful to retrieve then indexes)
        col_names = self.data.columns.values
        self.feat = [utils.clean_name(col_names[i]) for i in range(0, len(col_names)-1, 4)]

    def split_train_test(self, size = 0.2, prep = False):
        ''' if prep = True: fill NaN with median, drop duplicated rows'''
        x_tr, x_te, y_tr, y_te = train_test_split(self.data.iloc[:, :-1], self.data['target'],  test_size=size, random_state=42)
        
        if prep: 
            print('DT ANALYSIS...')
            # fill NaN
            x_tr, x_te = pre.fill_NaN(x_tr, x_te)
            
            # drop duplicated rows
            shape_before = x_tr.shape[0]
            x_tr.insert(x_tr.shape[1],'target', y_tr)
            x_tr.drop_duplicates(keep='first', inplace=True)

            print('Number of duplicated rows: {}\n'.format(shape_before - x_tr.shape[0]))
            
            y_tr = x_tr['target']
            x_tr = x_tr.iloc[:, :-1]
            
        self.set_train_test(x_tr, y_tr, x_te, y_te)

    def set_train_test(self, x_tr, y_tr, x_te, y_te):
        """Set train and test data"""
        self.train_dt = (x_tr, y_tr)
        self.test_dt = (x_te, y_te)    

    def get_train_test_sensors(self, sensors = []):
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
    
    def remove_sensor_feat(self, name):
        if name in self.feat:
            ind = self._get_i(name)
            self.data.drop(self.data.columns[list(range(ind, ind+4))], axis=1, inplace=True)
            self.feat.remove(name)

    def _get_i(self, name):
        ''' Get index of first sensor feature'''
        return self.feat.index(name) * 4

    
    
    
    