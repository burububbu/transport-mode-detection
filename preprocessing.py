
from sklearn import decomposition
import matplotlib.pyplot as plt
import numpy as np


def pca(x_tr, x_te): 
    pca = decomposition.PCA()
    pca.fit(x_tr)
    
    expl = pca.explained_variance_ratio_
    
    acc = 0
    i = 0
    for e in expl:
        acc = acc + e
        if acc >= 0.95:
            i = list(expl).index(e)  
            pca = decomposition.PCA(n_components = i)
            break
    print(i)
    pca.fit(x_tr)
    
    print(pca.explained_variance_ratio_)
    
    x_tr = pca.transform(x_tr)
    x_te = pca.transform(x_te)
    
    return (x_tr, x_te)
    

def standardization(x_train, x_test):
    mean = np.mean(x_train, axis=0)
    std = np.std(x_train, axis=0)
    x_train_st = (x_train - mean) / std
    x_test_st = (x_test - mean) / std
    return (x_train_st, x_test_st)

def min_max_scaling(x_train, x_test):
    max = np.max(x_train, axis=0)
    min = np.min(x_train, axis=0)

    x_train_st = (x_train - min) / (max - min)
    x_test_st = (x_test - min) / (max - min)

    return (x_train_st, x_test_st)


# Check how many nan values there are for each feature
def check_nan_sample(data, feat,  plot=False):
    tot_sample = data.shape[0]
    nan_sample = tot_sample - data.iloc[:, :-1].count()
    
    if plot:
        # multi bar plot to represent quantity of nan
        n = np.arange(0, 10) # 10 feat
        h = 0.25
        
        # 4 bars group
        plt.barh(n + 0.05, nan_sample[np.arange(0, 40, 4)], color='xkcd:azure', height=h-0.05) # blue mean
        plt.barh(n + 0.25, nan_sample[np.arange(1, 40, 4)], color='xkcd:light red', height=h) # red min
        plt.barh(n + 0.50, nan_sample[np.arange(2, 40, 4)], color='xkcd:medium green', height=h) # green max
        plt.barh(n + 0.75, nan_sample[np.arange(3, 40, 4)], color='xkcd:dull pink', height=h) # pink std
       
        plt.title('NaN values for each feature')
        plt.xlabel('number of Nan values')
        plt.ylabel('sensors')
        
        plt.yticks(np.arange(h + 0.1, 10), feat)
        plt.legend(['mean', 'min', 'max', 'std'])
       
    return nan_sample

def fill_NaN(data): # input x_train
    return data.fillna(data.median())

