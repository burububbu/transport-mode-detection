# -*- coding: utf-8 -*-
from tddataset import TDDataset
import pandas as pd

d = None
dt = None
path = '../dataset/dataset_5secondWindow.csv'
excluded_sensor = ['light', 'pressure', 'magnetic_field','gravity','proximity']
# and magnetic_field_uncalibrated
    
    # every sensor generates 4 features: mean, min, max, std
    # 64/4 = 16 sensors
    # but we have not to check this sensors (remove 6 x 4 features = 24):
        # light
	    # pression
        # magnetic field
    	# magnetic field uncalibrated
    	# gravity
    	# proximity
    # remaining features = 40


def main():
    global d
    global dt
    
    dtt = TDDataset()
    
    dtt.load_dataset_from_csv(path, excluded_sensor)
    
    d = dtt.get_data()
    print(d.columns.values)

if __name__ == '__main__':
    main()
    



