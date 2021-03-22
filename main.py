from models import knn_, random_forest_, svm_
from visualization import plot_multiple_acc, plot_train_results
import preprocessing as pre
from TDDataset import TDDataset
import numpy as np
import pandas as pd
import utils as utils
from cons import PATH, TO_EXCLUDE, TEST_SIZE, RF_ESTIMATORS, SVM_GRID, KNN_GRID, SVM_GRID


to_plot = {'nan_sample': 0, 'rf_each_s': 0, 'tot_results': 1}


def main():
    global final_results
    # load dataset
    dt = TDDataset(PATH, TO_EXCLUDE)

    # 1a. preprocessing
    pre.check_nan_sample(dt.data, dt.feat, plot= to_plot['nan_sample'])
    dt.remove_sensor_feat('step_counter')

    dt.split_train_test(TEST_SIZE, prep = True) # fill Nan with median, drop duplicates
    pre.check_balanced(dt.train_dt[1])

    # 2. search more discriminant sensor, returns results
    # s_accuracy = rf_each_sensor(dt, plot = to_plot['rf_each_s'])
    

    # 3a. select three subset sensor in base of that
    D1 = ['accelerometer','sound', 'orientation'] # the best
    D2 = ['accelerometer','sound', 'linear_acceleration']
    D3 = ['accelerometer','sound', 'gyroscope']

    
    D4 = ['accelerometer','sound','orientation','linear_acceleration','gyroscope', 'rotation_vector']
    D5 = dt.feat
    
    # 4. training different  classic model on each (cross validation when necessary)
    s_set = [D1, D2, D3, D4, D5]
    
    # models = ['lda']
    models = ['RF', 'SVM', 'KNN']
    # models = ['RF', 'SVM', 'KNN']

    final_results = pd.DataFrame(columns=models)
    
    for s in s_set:
        scores = []
        name = 'D' + str(s_set.index(s) + 1)

        print('Sensor set: {}: '.format(name))

        x_tr, x_te, y_tr, y_te = dt.get_train_test_sensors(s)
            
        # RANDOM FOREST
        if 'RF' in models:
            print('\t Creating RF classifier')
            _, score = random_forest_(x_tr, x_te, y_tr, y_te, n_estimators = RF_ESTIMATORS)
            scores.append(score)
        
        # SVM
        if 'SVM' in models:    
            print('\t Creating SVM classifier')
            x_tr_st, x_te_st = pre.standardization(x_tr, x_te)
            _, score =svm_(x_tr_st, x_te_st, y_tr, y_te, cv = True, param_grid = SVM_GRID)
            scores.append(score)
        
        #KNN
        if 'KNN' in models:
            print('\t Creating KNN classifier')
            x_tr_st, x_te_st = pre.standardization(x_tr, x_te)
            _, score = knn_(x_tr_st, x_te_st, y_tr, y_te, cv = True, param_grid = KNN_GRID)
            scores.append(score)
            
        # if 'lda':
        #    print('\t Creating SVM classifier')
        #    x_tr_st, x_te_st = pre.standardization(x_tr, x_te)
        #    n = len(s)*2
        #    print(n)
        #    x_tr, x_te = pre.lda_(x_tr_st, x_te_st, y_tr, n)
        #    _, score =svm_(x_tr_st, x_te_st, y_tr, y_te, cv = True, param_grid = SVM_GRID)
        #    scores.append(score)

        # NN
        if 'NN' in models:
            pass
           # _, score =  get_NN(name, s)
           # scores.append(score)

        final_results = final_results.append(pd.DataFrame([scores], columns = models, index=[name]))
        print('\n')
        

    # 3b. select, for each sensor the best feature and use for a subset of trainig
    # -

    # 5. plot results
    if to_plot['tot_results']:
        plot_train_results(' Final results', final_results, models, utils.s_names(len(s_set)))
        # plot_train_results(' Final results', final_results, models, utils.s_names(len(s_set)))
        # plot_train_results(' Final results', final_results, models, utils.s_names(len(s_set)))
    


def get_NN(set, s):
    pass

def rf_each_sensor(dt, plot = False):
    ''' With random forest check accuracy for single sensor model'''
    acc = []
    for name in dt.feat:
        data = dt.get_train_test_sensors([name])
        rf, score = random_forest_(*data, n_estimators= RF_ESTIMATORS)
        acc.append(score)
        rankVar = pd.Series(rf.feature_importances_, index=data[0].columns).sort_values(ascending=False)
        print(name, '\r', rankVar)
    
    sens_acc = pd.Series(acc, dt.feat)

    if plot:
        print(sens_acc.sort_values(ascending = False))
        plot_multiple_acc('Score for each sensor', sens_acc.index.values, sens_acc.values)
        
    return sens_acc

































if __name__ == '__main__':
    main()
    
    
    
    