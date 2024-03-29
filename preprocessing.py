import numpy as np

import visualization as vis

def standardize(x_train, x_test):
    """ Apply standardization on train and tets data"""
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train_st = (x_train - mean) / std
    x_test_st = (x_test - mean) / std

    return (x_train_st, x_test_st)


def fill_NaN(x_tr, x_te): 
    """ Fill NaN values with train median values."""
    median = x_tr.median()
    return x_tr.fillna(median), x_te.fillna(median)


def check_balanced(data):
    """ Print number of samples for each class"""
    uniq = np.unique(data, return_counts=True)

    print('Samples for each class:')
    for i in range(0, len(uniq[0])):
        print("\t{}: {} samples".format(uniq[0][i], uniq[1][i]))
    print('\n')


def check_nan_sample(data, feat):
    """ Return (and plot) number of NaN samples for each feature"""
    tot_sample = data.shape[0]
    nan_sample = tot_sample - data.iloc[:, :-1].count()  # series
    vis.plot_nan_values(nan_sample, feat)
    return nan_sample
