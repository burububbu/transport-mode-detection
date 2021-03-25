import matplotlib.pyplot as plt
import numpy as np

def plot_multiple_acc(title, x, h):
    ''' Plot multiple accuracies, on x axis the name of the feat ( or sensor)'''
    plt.title(title)  
    plt.bar(x, h)

    plt.yticks(np.arange(0, 1.10, step=0.1))
    plt.xticks(rotation=60)
    plt.show()
    
#             M1   M2   M3
#     index   
#      d1     5    5    6
#      d2    3     2     6
#      d3    1     5    3

def plot_train_results(title, df, models, sets):
    data = [df.loc[:, m] for m in models]

    x = np.arange(len(sets))
    n = 0.25 * (len(models) - 1) / 2 # first position
    v = 0

    _, ax = plt.subplots()
    rects = []
    
    for m in models:
        rects.append(ax.bar(x + v, data[models.index(m)], width=0.25, label=m))
        v = v + 0.25
    
    plt.title(title)
    plt.yticks(np.arange(0, 1.1, 0.1))
    plt.xticks(np.arange(n, len(sets), 1), sets)
    ax.legend(loc='lower right')

    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{:.2f}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 1),
                        textcoords="offset points",
                        ha='center', va='bottom')
    
    for rect_list in rects:
        autolabel(rect_list)

    plt.show()


def plot_nan_values(values, feat):
    ''' valori da rappresentare, nomi da mettere sull'asse delle y, raggruppamento di quante  bar, legenda'''
    # multi bar plot to represent quantity of nan
    n = np.arange(0, len(feat), 1.1) # 10 feat
    h = 0.25

    # 4 bars group
    plt.barh(n + 0.05, values[np.arange(0, 40, 4)], color='xkcd:azure', height=h, label='mean') # blue mean
    plt.barh(n + 0.25, values[np.arange(1, 40, 4)], color='xkcd:light red', height=h, label='min') # red min
    plt.barh(n + 0.50, values[np.arange(2, 40, 4)], color='xkcd:medium green', height=h, label='max') # green max
    plt.barh(n + 0.75, values[np.arange(3, 40, 4)], color='xkcd:dull pink', height=h, label='std') # pink std

    plt.title('NaN values for each feature')
    plt.xlabel('number of NaN values')
    plt.ylabel('sensors')

    # n = 0.25 * (len(sub feat) - 1) / 2
    n = 0.75/2 # first position

    plt.yticks(np.arange(n, 11, 1.1), feat)
    plt.legend()
    plt.show()
