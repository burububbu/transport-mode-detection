import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_acc(title, x, h):
    ''' Plot multiple accuracies, on x axis the name of the feat ( or sensor)'''
    plt.title(title)  
    plt.bar(x, h)
    plt.yticks(np.arange(0, 1.10, step=0.1))
    plt.xticks(rotation=60)
    

#             M1   M2   M3
#     index   
#      d1     5    5    6
#      d2    3     2     6
#      d3    1     5    3

# colonne = models, index = sets
def plot_train_results(title, df, models, sets):
    # sets = df.index.to_list()
    # models = df.columns.to_list()
    print(models)
    data = []

    for m in models:
        print(df.loc[:, m])
        data.append(df.loc[:, m])
    
    X = np.arange(len(sets))
    
    n = 0.25 * (len(models) - 1) / 2 # first position
    
    tot_bars = len(sets) * len(models)
    
    plt.xticks(np.arange(n, len(sets), 1), sets)
    v = 0

    for m in models:
        plt.bar(X + v, data[models.index(m)], width=0.25, label=m)
        v = v + 0.25
    plt.legend(loc= 'lower right')






# def plot_train_results(training_results, models, sets):
#     data = []

#     for m in models:
#         data.append(training_results.loc[training_results['model type'] == m].score)
    
#     X = np.arange(len(sets))
    
#     n = 0.25 * (len(models) - 1) / 2 # primo
    
#     tot_bars = len(sets) * len(models)
    
#     plt.xticks(np.arange(n, tot_bars* 0.25 + 0.01, 1), sets)
#     v = 0
#     for m in models:
#         plt.bar(X + v, data[models.index(m)], width=0.25, label=m)
#         v = v + 0.25
#     plt.legend(loc= 'lower right')


    # for m in ['RF', 'KNN', 'SVM']: # SOSTITUIRE CON NèP.UNIQUE
    #     data.append(training_results.loc[training_results['model type'] == m].score)
    
    # X = np.arange(3)
    # plt.xticks(np.arange(0.25, 2.26, 1), ['D0','D1','D2'])
    # plt.bar(X + 0.00, data[0], color='b', width=0.25, label='RF')
    # plt.bar(X + 0.25, data[1], color='g', width=0.25, label='KNN')
    # plt.bar(X + 0.50, data[2], color='r', width=0.25, label='SVM')
    # plt.legend()