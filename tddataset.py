# TODO
class TDDataset:
    """Transport detection dataset with sensor data"""
    def __init__(self):
        self.data = None
        self.target = None
    
    def load_dataset_from_csv(self, csv_path, excluded_sensors=None):
        """Load dataset from cvs file"""
        pass
    
    def get_dataset(self):
        """Get entire dataset"""    
        return self.data
    
    def get_dataset_features(self, features):
        """ Get dataset with specific features"""
        pass

    def get_target(self):
        """ Get target data"""
        pass
        
    