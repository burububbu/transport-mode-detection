
from os import remove
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

results_columns = ['sensors', 'model type', 'score']
training_results = pd.DataFrame(columns= results_columns)

classifiers = {
    'RF': models.random_forest_,
    'SVM': models.svm_,
    'KNN': models.knn_
}

def main():
    global sensors_set

    dt = TDDataset(path, excluded_sensor)
    
    # first preprocessing 
    first_preprocessing(dt) # on entire training dataset + test dataset
    
    # for a first investigation of how the sensors can be discriminating with the defined classes, we choose random forest algorithm.
   # check_each_sensor(dt, plot=False)

    # assign best sensors (based on rf_each_sensor output)
    sensor_set =  [['accelerometer','sound', 'orientation'], # 12 feat
    ['accelerometer','sound','orientation','linear_acceleration','gyroscope', 'rotation_vector'], # 24 feat
     dt.feat] #36 feat
 
    # TEST on the three datasets // RANDOM FOREST, SVM, NB, KNN
    for s in sensor_set:
        name = 'D' + str(sensor_set.index(s))
        print('Sensors, index: ', name, s)
        data = dt.get_train_test_feat(s)

        # RANDOM FOREST
        get_classifier(*data, name, 'RF', {'n_estimators': 200}) 

        # KNN
        params= {'n_neighbors': np.arange(2, 10, 2)}
        get_classifier(*data, name, 'KNN', {'cv' : True, 'param_grid' : params}, stand = True )

        # # SVM
        params = {'C': np.arange(200, 800, step=200)}
        get_classifier(*data, name, 'SVM', {'cv' : True, 'param_grid' : params}, stand = True )

        print()

    # here for the neural net


    plot_train_results(training_results)

def get_classifier(x_tr, x_te, y_tr, y_te, sensors, type, settings, stand = False):
    '''get classifier'''
    if stand:
         x_tr, x_te = prep.standardization(x_tr, x_te)

    print('\t Creating ', type, 'classifier')
    res = classifiers[type](x_tr, x_te, y_tr, y_te, settings)
    update_res(sensors, type, res[1])


def update_res(s, type, score):
    global training_results
    training_results = training_results.append(
    pd.Series([s, type, score], index = results_columns), ignore_index=True)

def check_each_sensor(dt, plot= False):
    ''' With random forest check accuracy for single sensor model'''
    acc = []
    for name in dt.feat:
        data = dt.get_train_test_feat([name])
        res = models.random_forest_(*data, settings = {'n_estimators': 500})
        acc.append(res[1])
    
    sens_acc = pd.Series(acc, dt.feat)
    
    if plot:
        plt.title('accuracy single sensor')  
        plt.bar(dt.feat, acc)
        plt.yticks(np.arange(0, 1.10, step=0.1))
        plt.xticks(rotation=60)
        
    return sens_acc

def first_preprocessing(dt, plot=False):
    global uniq
    global nan_samples
    global medians
    global describe
    # 1. check NaN values
    nan_samples = prep.check_nan_sample(dt.data, dt.feat, plot=plot)
    dt.remove_sensor_feat('step_counter')
    
    # 2. split training and test and analyze training set
    df = dt.data
    x_train, x_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'],  test_size=0.2, random_state=42)
    
    # fill NaN
    x_train, x_test = prep.fill_NaN(x_train, x_test)

    # drop duplicated rows
    shape_before = x_train.shape[0]
    x_train.insert(x_train.shape[1],'target', y_train)
    x_train.drop_duplicates(keep='first', inplace=True)
    print('Number of duplicated rows: ', shape_before - x_train.shape[0])
    
    y_train = x_train['target']
    x_train = x_train.iloc[:, :-1]
    
    # check balanced training dataset
    uniq = pd.DataFrame(np.unique(y_train, return_counts=True)) # è bilanciato 

    dt.set_train_test(x_train, y_train, x_test, y_test)

def plot_train_results(res):
    data = []
    for m in ['RF', 'KNN']:
        data.append(training_results.loc[training_results['model type'] == m].score)
    
    print(data)
    X = np.arange(3)
    plt.xticks(np.arange(0.25, 2.26, 1), ['D0','D1','D2'])
    plt.bar(X + 0.00, data[0], color='b', width=0.25, label='RF')
    plt.bar(X + 0.25, data[1], color='g', width=0.25, label='KNN')
    plt.legend()
       



if __name__ == '__main__':
    main()
    
    
    
    

    