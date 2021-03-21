import re

def clean_name(n):
    ''' clean features name'''
    return re.search('(.+?)[#]', n).group(1).replace('android.sensor.', '')

def s_names(l):
    return ['D' + str(i) for i in range(1, l+1)]
    
# def plot_dist(data, i):
#     feat = data.iloc[:, i].dropna()
#     plt.hist(feat, bins = np.arange(0, feat.max(), 0.05 ))

