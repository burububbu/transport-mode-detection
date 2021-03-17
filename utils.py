# from sklearn.model_selection import train_test_split
import re
import numpy as np
import matplotlib.pyplot as plt

def clean_name(n):
    ''' clean features name'''
    return re.search('(.+?)[#]', n).group(1).replace('android.sensor.', '')

    
def plot_dist(data, i):
    feat = data.iloc[:, i].dropna()
    plt.hist(feat, bins = np.arange(0, feat.max(), 0.05 ))

