import pandas as pd
import re
# TODO
class TDDataset:
    """
    Transport detection dataset with sensor data.
    Fields:
        - feat
        - data
        - target
        -                
    """
    def __init__(self, csv_path, excluded_sensors=[]):
        feat_list = self._analyze_feat(csv_path, excluded_sensors)
        self.data, self.target = self._load_dataset_from_csv(csv_path, feat_list)
        
    def _load_dataset_from_csv(self, csv_path, feat_list):
        """Load dataset from cvs file"""
        data = pd.read_csv(csv_path, usecols = feat_list)
        target = pd.read_csv(csv_path, sep=',', usecols = ['target'])
        
        return data, target
    
    def get_data_feat(self, sensors = []):
        """Get entire dataset or, if specified, return specific columns"""
        indexes = []
        
        for s in sensors:
            ind = self._feat.index(s) * 4
            indexes.extend(list(range(ind, ind + 4)))
        
        return self.data.iloc[:, indexes]
    
    def _analyze_feat(self, csv_path, excluded_sensors):
        """create list of names to exclude"""
        feat_names = pd.read_csv(csv_path, header=None, nrows=1).iloc[:, 5:-2]
        # for t in temp.values[0]:
        #     for s in excluded_sensors:
        #         if s in t:
        #             to_exclude.append(t)
        
        feat_list= [f for f in feat_names.values[0] if f not in
                    [t for s in excluded_sensors for t in feat_names.values[0] if s in t]]
        
        # lista solo con i nomi delle feat
        self._feat = [get_cleaned_name(feat_list[i]) for i in range(0, len(feat_list), 4)]
        
        return feat_list
        
def get_cleaned_name(name):
    return re.search('(.+?)[#]', name).group(1).replace('android.sensor.', '')
    
    