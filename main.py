from sklearn.preprocessing import LabelEncoder
from models import knn_, random_forest_, svm_
import visualization as vis
import preprocessing as pre
from TDDataset import TDDataset
import pandas as pd
from cons import *
import nn

to_plot = {
    'nan_sample': 0,
    'rf_each_s': 0,
    'loss_values': 0,
    'tot_results': 1
    }

def main():
    global final_results # remove

    # load dataset
    dt = TDDataset(PATH, TO_EXCLUDE)

    # 1a. preprocessing
    pre.check_nan_sample(dt.data, dt.feat, plot=to_plot['nan_sample'])
    dt.remove_sensor_feat('step_counter')

    dt.split_train_test(TEST_SIZE, prep = True) # fill Nan with median, drop duplicates
    pre.check_balanced(dt.train_dt[1])

    # 2. search more discriminant sensor, returns results
    # s_accuracy = rf_each_sensor(dt, plot = to_plot['rf_each_s'])
    # print('RF accuracy for each sensor')
    # print('\n', s_accuracy.sort_values(ascending=False))
    
    # 3a. select three subset sensor in base of accuracy
    D1 = ['accelerometer','sound', 'orientation'] # the best
    D2 = ['accelerometer','sound', 'linear_acceleration']
    D3 = ['accelerometer','sound', 'gyroscope']

    D4 = ['accelerometer','sound','orientation','linear_acceleration','gyroscope', 'rotation_vector']
    # D5 = all feat
    
    # 4. training different models on each (cross validation when necessary)
    s_set = [
        {'name': 'D1', 'sensors': D1},
        {'name': 'D4', 'sensors': D4},
        {'name': 'D5', 'sensors': dt.feat}
        ]
    
    # models i want to train
    models_order = ['NN']
    # models_order = ['RF', 'SVM', 'KNN', 'NN']

    final_results = pd.DataFrame(columns=models_order) # df with final results

    # train different models
    for s in s_set:
        scores = [] # score for each model
        
        x_tr, x_te, y_tr, y_te = dt.get_train_test_sensors(s['sensors'])

        print('Sensor set: {}: '.format(s['name']))
        for m in models_order:

            print('\t Training {} classifier'.format(m))
            
            _, score = get_classifier(m, s['name'], x_tr, x_te, y_tr, y_te)
            scores.append(score)
        
        print('\n')
        
        final_results = final_results.append(pd.DataFrame([scores], columns = models_order, index=[s['name']]))

    # 5. plot results
    if to_plot['tot_results']:
        vis.plot_train_results('Final results', final_results, models_order, [s['name'] for s in s_set])
        
# change this, dict si crea solo per modelli che scelgo
def get_classifier(m, s, x_tr, x_te, y_tr, y_te):
    to_ret = None
    if m == 'RF':
        to_ret = random_forest_(x_tr, x_te, y_tr, y_te, n_estimators = RF_ESTIMATORS)

    elif m == 'SVM':
        x_st = pre.standardize(x_tr, x_te)
        to_ret = svm_(*x_st, y_tr, y_te, cv = TO_VAL, param_grid = SVM_GRID, c=C[s])

    elif m == 'KNN':
        x_st = pre.standardize(x_tr, x_te)
        to_ret= knn_(*x_st, y_tr, y_te, cv = TO_VAL, param_grid = KNN_GRID, neighbors=N_NEIGHBORS[s])

    elif m == 'NN':
        # encode the label
        le = LabelEncoder()
        le.fit(y_tr)

        y_tr_enc = le.transform(y_tr) 
        y_te_enc = le.transform(y_te)

        # standardizzazione
        x_st = pre.standardize(x_tr, x_te)

        to_ret = nn.get_nn(
            *x_st,
            y_tr_enc, y_te_enc,
            v = NN_TO_VAL,
            hs = HIDDEN_SIZE[s],
            lr = L_RATE[s],
            epochs = EPOCHS[s],
            bs = BATCH_SIZE[s],
            drop= DROPOUT[s] )

    return to_ret

def rf_each_sensor(dt, plot = False):
    ''' With random forest check accuracy for single sensor model'''
    acc = []
    for name in dt.feat:
        data = dt.get_train_test_sensors([name])
        rf, score = random_forest_(*data, n_estimators= RF_ESTIMATORS)
        acc.append(score)
        
        # rankVar = pd.Series(rf.feature_importances_, index=data[0].columns).sort_values(ascending=False)
        # print(name, '\r', rankVar)
    
    sens_acc = pd.Series(acc, dt.feat)

    if plot:
        vis.plot_multiple_acc('Score for each sensor', sens_acc.index.values, sens_acc.values)
        
    return sens_acc


if __name__ == '__main__':
    main()