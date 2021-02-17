import pandas as pd
# TODO
class TDDataset:
    """Transport detection dataset with sensor data"""
    def __init__(self):
        self.data = None
        self.target = None
    
    def load_dataset_from_csv(self, csv_path, excluded_sensors=[]):
        """Load dataset from cvs file"""
        to_exclude = []
        
        if excluded_sensors:
            to_exclude = self._analyze_feat(csv_path, excluded_sensors)
        
        self.data = pd.read_csv(csv_path, usecols = lambda n: n not in to_exclude).iloc[:, 5:-2]
        
        self.target = pd.read_csv(csv_path, sep=',', usecols = ['target'])
        
    def get_data(self, features=[]):
        """Get entire dataset or, if specified, return specific columns"""
        # TODO
        return self.data

    def get_target(self):
        """ Get target data"""
        return self.target
    
    def _analyze_feat(self, csv_path, excluded_sensors):
        """ create list of names to exclude"""
        temp = pd.read_csv(csv_path, header=None, nrows=1).iloc[:, 5:-2]
        # for t in temp.values[0]:
        #     for s in excluded_sensors:
        #         if s in t:
        #             to_exclude.append(t)
        return [t for s in excluded_sensors for t in temp.values[0] if s in t]
        
                    
    