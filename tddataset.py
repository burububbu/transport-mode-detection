import pandas as pd
import utils 

class TDDataset:
    """
    Transport detection dataset with sensor data.
    Fields:
        - features
        - data
    """
    def __init__(self, csv_path, excluded_sensors=[]):
        self.feat, feat_list = self._analyze_feat(csv_path, excluded_sensors)
        self.data = pd.read_csv(csv_path, usecols = feat_list)

    def get_data_feat(self, sensors = []):
        """Get entire dataset or, if specified, return specific columns"""
        indexes = []
        
        if sensors:
            for s in sensors:
                ind = self.feat.index(s) * 4
                indexes.extend(list(range(ind, ind + 4)))
            indexes.append(self.data.columns.get_loc('target'))
        
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
           
        feat_cleaned = [utils.clean_name(feat_list[i]) for i in range(0, len(feat_list), 4)]
        
        feat_list.append('target')
        
        return feat_cleaned, feat_list

        

    
    