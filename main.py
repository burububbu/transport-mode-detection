
from TDDataset import TDDataset
import visualization as vis
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
import preprocessing as pre
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
    global d

    dt = TDDataset(path, excluded_sensor)
    
    # initial check NaN values, remove step_counter (too much nan values)
    nan_samples = pre.check_nan_sample(dt.data, dt.feat, plot=False)
    dt.remove_sensor_feat('step_counter')
    
    # split dataset, if prep = True, fill Nan, check duplicates and check balanced dataset
    dt.split_train_set(prep = True)
    
    d = dt.train_dt
    
    # for a first investigation of how the sensors can be discriminating with the defined classes, we choose random forest algorithm.
   # check_each_sensor(dt, plot=False)

    # assign best sensors (based on rf_each_sensor output)
    sensor_set =  [['accelerometer','sound', 'orientation'], # 12 feat
    ['accelerometer','sound','orientation','linear_acceleration','gyroscope', 'rotation_vector'], # 24 feat
     dt.feat] #36 feat
 
    print('\nMODELING...')
    # TEST on the three datasets // RANDOM FOREST, SVM, NB, KNN
    for s in sensor_set:
        name = 'D' + str(sensor_set.index(s))
        print('Sensors groups: {}: '.format(name))
        data = dt.get_train_test_feat(s)

        # RANDOM FOREST
        get_classifier(*data, name, 'RF', {'n_estimators': 200}) 

        # KNN
        params= {'n_neighbors': np.arange(2, 10, 2)}
        get_classifier(*data, name, 'KNN', {'cv' : False, 'param_grid' : params}, stand = True )

        # SVM
        params = {'C': np.arange(200, 800, step=200)}
        get_classifier(*data, name, 'SVM', {'cv' : False, 'param_grid' : params}, stand = True )
        
        print('\n')

    # here for the neural net


    vis.plot_train_results(training_results, ['RF', 'KNN', 'SVM'], ['D0', 'D1', 'D2'])

def get_classifier(x_tr, x_te, y_tr, y_te, sensors, type, settings, stand = False):
    '''get classifier'''
    if stand:
         x_tr, x_te = pre.standardization(x_tr, x_te)

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

if __name__ == '__main__':
    main()
    
    
    
    

    