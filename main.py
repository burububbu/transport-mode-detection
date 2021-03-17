
from TDDataset import TDDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import preprocessing as prep
import models as models


    
path = '../dataset/dataset_5secondWindow.csv'
excluded_sensor = ['light', 'pressure', 'magnetic_field','gravity','proximity']
medians = [] #save here median values to use
# i need it when i have to fill missing value in new raw data (model input)

miao = []
results_columns = ['sensors', 'model type', 'score']
results = pd.DataFrame(columns= results_columns)

D1_s = None
D2_s = None
D3_s = None

def main():
    global D1_s, D2_s, D3_s
    dt = TDDataset(path, excluded_sensor)
    
    first_preprocessing(dt) # on entire training dataset + test dataset
   # rf_each_sensor(dt, plot=True)
    
    # assign best sensors (based on rf_each_sensor output)
    D1_s = ['accelerometer','sound','orientation'] # 12 feat
    D2_s = ['accelerometer','sound','orientation','linear_acceleration','gyroscope', 'rotation_vector'] # 20 feat
    D3_s = dt.feat #36 feat

    # TEST on the three datasets // RANDOM FOREST, SVM, NB, KNN, NEURAL NET
    for s in [D1_s, D2_s, D3_s]:
        data = dt.get_train_test_feat(s)
        get_random_forest(*data, s) # RANDOM FOREST
        get_SVM(*data, s) # SVM 
        # get_bt(*data, s) # boosting tree
        get_knn(*data, s) # KNN
     

def get_knn(x_tr, x_te, y_tr, y_te, sensors):
    global miao
    # x_tr, x_te = prep.min_max_scaling(x_tr, x_te)
    x_tr, x_te = prep.standardization(x_tr, x_te)
    params = {'n_neighbors': np.arange(2, 10, 2)}
    res = models.knn_(x_tr, x_te, y_tr, y_te, cv = True, param_grid= params)
    # update results
    update_res(pd.Series([sensors, 'KNN', res[1]], index = results_columns))
    miao.append(res[2])
     

# use separate function to record the results
def get_random_forest(x_tr, x_te, y_tr, y_te, sensors):
    res = models.random_forest_(x_tr, x_te, y_tr, y_te, est = 200)
    # update results
    update_res(pd.Series([sensors, 'RF', res[1]], index = results_columns))


def get_SVM(x_tr, x_te, y_tr, y_te, sensors):
    # STANDARDIZATION
    x_train, x_test = prep.standardization(x_tr, x_te)
    # search for model and cross validation
    params = {'C': np.arange(200, 800, step=200)}
    res = models.svm_(x_train, x_test, y_tr, y_te, cv = True, param_grid = params)
    update_res(pd.Series([sensors, 'SVM', res[1]], index = results_columns))

def get_bt(x_tr, x_te, y_tr, y_te, sensors):
    res = models.boosting_tree_(x_tr, x_te, y_tr, y_te, est=500)
    update_res(pd.Series([sensors, 'Boosting tree', res[1]], index = results_columns))


def update_res(se):
    global results
    results = results.append(se, ignore_index=True)


def rf_each_sensor(dt, plot= True):
    ''' With random forest check accuracy for single sensor model'''
    global sensors_accuracy # acc, sound, orientation and linear acc are the best
    acc = []
    for name in dt.feat:
        data = dt.get_train_test_feat([name])
        res = models.random_forest(*data, est=500)
        acc.append(res[1])
        
    sensors_accuracy = pd.Series(acc, dt.feat)
    
    if plot:
        plt.title('accuracy single sensor')  
        plt.bar(dt.feat, acc)
        plt.yticks(np.arange(0, 1.10, step=0.1))
        plt.xticks(rotation=60)



def first_preprocessing(dt):
    global uniq
    global nan_samples
    global medians
    global describe
    # PREP on entire dataset
    # 1. check NaN values -----------------------------------------------------
    nan_samples = prep.check_nan_sample(dt.data, dt.feat, plot=False)
    # remove step counter bc there are many NaN
    # doubts also on speed features (specially #std)
    
    # tot sample: 5893
    dt.remove_sensor_feat('step_counter')
    
    # 2. split training and test and analysis on training set
    df = dt.data
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'],  test_size=0.2, random_state=42 )
    # d = x_train.to_numpy()
    describe = df.describe()
    # PREP only on training
    # fill Nan values -> median
    # check speed#std distribution (bc there are many missing values)
    
    x_train = prep.fill_NaN(x_train)
    medians = x_train.median()
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
    x_test = prep.fill_NaN(x_test)
    
    # for a first investigation of how the sensors can be discriminating with the defined classes, we choose random forest algorithm.
    
    dt.set_train_test(x_train, y_train, x_test, y_test)
    
    # --- finished first preprocessing

    
if __name__ == '__main__':
    main()
    
    
    
    

    