
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import nn
import cons as c
import models as model
import visualization as vis
import preprocessing as pre
from TDDataset import TDDataset

def get_classifier(m, s, x_tr, x_te, y_tr, y_te):
    """ Train a classifier.
        Cross-validate if const TO_VAL=True and/or NN_TO_VAL=False.

        Params:
            - m: {‘RF’, ‘SVM’, ‘KNN’, ‘NN’}
            - s: {'D1', 'D4', 'D5'}
            - x_tr, x_te, y_tr, y_te: training data, test data

        Returns: (classifier instance, score) 
    """
    to_ret = None
    if m == 'RF':
        to_ret = model.random_forest_(x_tr, x_te, y_tr, y_te, n_estimators=c.RF_ESTIMATORS)

    elif m == 'SVM':
        x_st = pre.standardize(x_tr, x_te)
        to_ret = model.svm_(*x_st, y_tr, y_te, cv=c.TO_VAL, param_grid=c.SVM_GRID, c= c.C[s])

    elif m == 'KNN':
        x_st = pre.standardize(x_tr, x_te)
        to_ret = model.knn_(*x_st, y_tr, y_te, cv=c.TO_VAL, param_grid=c.KNN_GRID, neighbors=c.N_NEIGHBORS[s])

    elif m == 'NN':
        le = LabelEncoder()
        le.fit(y_tr)

        y_tr_enc = le.transform(y_tr)
        y_te_enc = le.transform(y_te)

        x_st = pre.standardize(x_tr, x_te)

        to_ret = nn.get_nn(
            *x_st,
            y_tr_enc, y_te_enc,
            v=c.NN_TO_VAL,
            hs=c.HIDDEN_SIZE[s],
            lr=c.L_RATE[s],
            ep=c.EPOCHS[s],
            bs=c.BATCH_SIZE[s],
            drop=c.DROPOUT[s])

    return to_ret


def rf_each_sensor(dt, plot=False):
    ''' Get accuracy for each sensor (4 features) with random forest model.'''
    acc = []
    for name in dt.feat:
        data = dt.get_train_test_sensors([name])
        _, score = model.random_forest_(*data, n_estimators=c.RF_ESTIMATORS)
        acc.append(score)

        # rankVar = pd.Series(rf.feature_importances_, index=data[0].columns).sort_values(ascending=False)
        # print(name, '\r', rankVar)

    sens_acc = pd.Series(acc, dt.feat)

    if plot:
        vis.plot_multiple_acc('Score for each sensor', sens_acc.index.values, sens_acc.values)

    return sens_acc


if __name__ == '__main__':
    # start analysis process
    to_plot = {
        'nan_sample': 0,
        'rf_each_s': 0,
        'loss_values': 0,
        'tot_results': 1
    }

    dt = TDDataset(c.PATH, c.TO_EXCLUDE)

    # preprocessing phase
    pre.check_nan_sample(dt.data, dt.feat, plot=to_plot['nan_sample'])
    dt.remove_sensor_feat('step_counter')

    dt.split_train_test(c.TEST_SIZE, prep=True)  # fill Nan with median, drop duplicates
    pre.check_balanced(dt.train_dt[1])

    # search for more discriminant sensor with random forest
    s_accuracy = rf_each_sensor(dt, plot = to_plot['rf_each_s'])
    print('RF accuracy for each sensor')
    print('\n', s_accuracy.sort_values(ascending=False))

    # select three subset sensor based on RF accuracy
    # D1 = ['accelerometer', 'sound', 'orientation']  # the best
    # D2 = ['accelerometer', 'sound', 'linear_acceleration']
    # D3 = ['accelerometer', 'sound', 'gyroscope']

    s_set = [
            {'name': 'D1',
            'sensors': ['accelerometer', 'sound', 'orientation']
            },
            
            {'name': 'D4',
            'sensors': ['accelerometer', 'sound', 'orientation', 'linear_acceleration', 'gyroscope', 'rotation_vector']
            },
            
            {'name': 'D5',
            'sensors': dt.feat}
            ]

    # train different models on each set (with cross validation if necessary)
    models_order = ['NN'] 
    # models_order = ['RF', 'SVM', 'KNN', 'NN']  # models to train

    final_results = pd.DataFrame(columns=models_order)

    for s in s_set:
        scores = []
        x_tr, x_te, y_tr, y_te = dt.get_train_test_sensors(s['sensors'])

        print('Sensor set {}: {} '.format(s['name'], s['sensors']))
        for m in models_order:
            print('\t Training {} classifier'.format(m))
            _, score = get_classifier(m, s['name'], x_tr, x_te, y_tr, y_te)
            scores.append(score)
        print('\n')

        final_results = final_results.append(pd.DataFrame([scores], columns=models_order, index=[s['name']]))

    # plot results
    if to_plot['tot_results']:
        vis.plot_train_results('Final results', final_results, models_order, [s['name'] for s in s_set])
