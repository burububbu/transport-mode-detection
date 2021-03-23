import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

def min_max_scaling(x_train, x_test):
    max = np.max(x_train, axis=0)
    min = np.min(x_train, axis=0)

    x_train_st = (x_train - min) / (max - min)
    x_test_st = (x_test - min) / (max - min)

    return (x_train_st, x_test_st)


def lda_(x_tr, x_te, y_tr, n):
    lda = LinearDiscriminantAnalysis(n_components=2)
    lda.fit(x_tr, y_tr)
    x_tr_t = lda.transform(x_tr)
    x_te_t = lda.transform(x_te)
    return x_tr_t, x_te_t

def standardization(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)

    x_train_st = (x_train - mean) / std
    x_test_st = (x_test - mean) / std

    return (x_train_st, x_test_st)

def fill_NaN(x_tr, x_te): # input x_train
    median = x_tr.median()
    return x_tr.fillna(median), x_te.fillna(median) 

def check_balanced(data):
    uniq = np.unique(data, return_counts=True) # è bilanciato
    print('\rSamples for each class\r')
    for i in range(0, len(uniq[0])):
        print("\t{}: {} samples".format(uniq[0][i], uniq[1][i]))
  
# Check how many nan values there are for each feature
def check_nan_sample(data, feat, plot=False):
    tot_sample = data.shape[0]
    nan_sample = tot_sample - data.iloc[:, :-1].count()
    
    
    # if plot:
    #     # multi bar plot to represent quantity of nan
    #     n = np.arange(0, 10) # 10 feat
    #     h = 0.25
        
    #     # 4 bars group
    #     plt.barh(n + 0.05, nan_sample[np.arange(0, 40, 4)], color='xkcd:azure', height=h-0.05) # blue mean
    #     plt.barh(n + 0.25, nan_sample[np.arange(1, 40, 4)], color='xkcd:light red', height=h) # red min
    #     plt.barh(n + 0.50, nan_sample[np.arange(2, 40, 4)], color='xkcd:medium green', height=h) # green max
    #     plt.barh(n + 0.75, nan_sample[np.arange(3, 40, 4)], color='xkcd:dull pink', height=h) # pink std
       
    #     plt.title('NaN values for each feature')
    #     plt.xlabel('number of Nan values')
    #     plt.ylabel('sensors')
        
    #     plt.yticks(np.arange(h + 0.1, 10), feat)
    #     plt.legend(['mean', 'min', 'max', 'std'])
       
    return nan_sample




