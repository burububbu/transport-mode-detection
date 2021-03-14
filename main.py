from TDDataset import TDDataset
import utils
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import pandas as pd 
import preprocessing as prep
    

path = '../dataset/dataset_5secondWindow.csv'
excluded_sensor = ['light', 'pressure', 'magnetic_field','gravity','proximity']
medians = [] #save here median values to use
# i need it when i have to fill missing value in new raw data (model input)

def main():
    global miao
    
    dt = TDDataset(path, excluded_sensor)
    
    first_preprocessing(dt) # on entire training dataset + test dataset


def first_preprocessing(dt):
    # PREP on entire dataset
    # 1. check NaN values -----------------------------------------------------
    nan_sample = prep.checkSampleNotNull(dt.data, dt.feat, plot=True)
    # remove step counter bc there are many NaN
    # doubts also on speed features (specially #std)
    
    # tot sample: 5893
    dt.remove_sensor_feat('step_counter')
    
    # 2. split training and test and analysis on training set
    df = dt.data
    
    x_train, x_test, y_train, y_test = utils.split_dataset(df.iloc[:, :-1], df['target'])
    # d = x_train.to_numpy()
    
    # PREP only on training
    # fill Nan values -> median
    # check speed#std distribution (bc there are many missing values)
    # plot_distribution('speed', 3) # skewed, use median here
    
    x_train = fill_NaN(x_train)
    # CHECK SAMPLES DUPLICATED (after filling)
    
    shape_before = x_train.shape[0]
    # necessario perchè devo eliminare anche le y corrispondenti
    x_train.insert(x_train.shape[1],'target', y_train)
    # first -> keep the first element
    x_train.drop_duplicates(keep='first', inplace=True)
    
    print('Number of duplicated rows: ', shape_before - x_train.shape[0])
    
    y_train = x_train['target']
    x_train = x_train.iloc[:, :-1]
    
    # CHECK BALANCED DATASET
    uniq = pd.DataFrame(np.unique(y_train, return_counts=True)) # è bilanciato
    
    # assuming each feature has skewed distribution

    # describe = x_train.describe()
    
    # fill nan values on test with train values (i have to suppose that i don't
    # know anything about test, i can't retrieve information from that)
    x_test = fill_NaN(x_test)
    
    # for a first investigation of how the sensors can be discriminating with the defined classes, we choose random forest algorithm.
    
    dt.set_train_test(x_train, y_train, x_test, y_test)
    
    # --- finished first preprocessing
    
    x_train_acc, _ = dt.get_train_test_feat(['gyroscope', 'speed'])
  
    
    for s in range(0, len(dt.feat)):
        ind = s*4
        # rf(x_train.iloc[:, ind: ind+4 ], y_train, x_test.iloc[:, ind: ind+4 ], y_test)
    # ex_rf(x_train, y_train, x_test, y_test)
    


def rf(x, y, x_t, y_t):
    rf = DecisionTreeClassifier(max_depth=8)
    # rf = RandomForestClassifier(n_estimators=800)
    rf.fit(x, y)
    print(rf.score(x_t, y_t))

# def plot_distribution(sensor, i):
#     if dt:
#         feat = dt.get_data_feat([sensor]).iloc[:, i].dropna()
#         print('mean', np.mean(feat))
#         print('median', np.median(feat))
#         plt.hist(feat, bins = np.arange(0, feat.max(), 0.05 ))

def fill_NaN(data): # input x_train
    return data.fillna(data.median())
    
if __name__ == '__main__':
    main()
    
    
    
    

    