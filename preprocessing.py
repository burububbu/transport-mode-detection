import matplotlib.pyplot as plt
import numpy as np

# Check how many nan values there are for each feature
def checkSampleNotNull(data, feat,  plot=False):
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
