# -*- coding: utf-8 -*-
from tddataset import TDDataset

df = None
dt = None
path = '../dataset/dataset_5secondWindow.csv'
excluded_sensor = ['light', 'pressure', 'magnetic_field','gravity','proximity']
# and magnetic_field_uncalibrated

#features
# 	accelerometer,
# 	game_rotation_vector,
# 	gravity,
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
    global df
    global dt
    # dataset instance
    dt = TDDataset(path, excluded_sensor)
    random_forest()
    
    
def random_forest():
    global df
    global dt
    df = dt.get_data_feat(['accelerometer', 'gyroscope'])
    RF_preprocessing()


# serie of functions to handle different types of prprocessing based on the type of model
def RF_preprocessing():
    global df
    df.dropna(inplace=True)
    print(df.head().describe())
    
if __name__ == '__main__':
    main()
    



