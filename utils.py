from sklearn.model_selection import train_test_split
import re

def clean_name(n):
    ''' clean feat name'''
    return re.search('(.+?)[#]', n).group(1).replace('android.sensor.', '')

def split_dataset(data, target, val=False):
    ''' val = True if u want also the val set'''
    x_tr, x_te, y_tr, y_te = train_test_split(data, target, test_size=0.2, random_state=42)
    if val:
        # split training set
       x_tr, x_va, y_tr, y_va = train_test_split(x_tr, y_tr, test_size=0.25, random_state=42)
       return x_tr, x_va, x_te, y_tr, y_va, y_te
    return x_tr, x_te, y_tr, y_te