# from sklearn.model_selection import train_test_split
import re

def clean_name(n):
    ''' clean feat name'''
    return re.search('(.+?)[#]', n).group(1).replace('android.sensor.', '')