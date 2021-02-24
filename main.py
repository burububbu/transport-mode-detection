# -*- coding: utf-8 -*-

from tddataset import TDDataset
import preprocessing as pr

df = None
dt = None
d = None
path = '../dataset/dataset_5secondWindow.csv'
excluded_sensor = ['light', 'pressure', 'magnetic_field','gravity','proximity']
# and magnetic_field_uncalibrated

#features
# 	accelerometer,
# 	game_rotation_vector,
# 	gyroscope,
# 	gyroscope_uncalibrated,
# 	linear_acceleration,
# 	orientation,
# 	rotation_vector,
# 	step_counter,
# 	sound,
# 	speed,
    
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
    global dt, df, d
    # dataset instance
    dt = TDDataset(path, excluded_sensor)
    d = dt.data
    # TODO fare il filling dei NaN con un valore
    
    # 1. check sensor with more information -> create logistic regression for every sensor and check accuracy
    #check_sensor_importance(dt)
    RF_preprocessing()

    

def check_sensor_importance(dt):
    # use logistic regression
    for s in dt.feat:
        dt.get_data_feat([s])
          
# serie of functions to handle different types of prprocessing based on the type of model
def RF_preprocessing():
    global d
    pr.check_balanced_dt(d)
    # df.dropna(inplace=True)
    # print(df.head().describe())


if __name__ == '__main__':
    main()
    



