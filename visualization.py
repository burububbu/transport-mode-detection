import numpy as np
import matplotlib.pyplot as plt

def plot_multiple_acc(title, x, h):
    """ A simple bar plot """ # sensor names on x axis
    plt.title(title)  
    plt.bar(x, h)

    plt.yticks(np.arange(0, 1.10, step=0.1))
    plt.xticks(rotation=60)
    plt.show()
    

def plot_train_results(title, df, models, sets):
    """ Plot grouped bar plot for final training results representation"""

# df structure:
#             M1  M2  M3
#     index   
#      d1     -   -   -  
#      d2     -   -   -  
#      d3     -   -   -

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

    def autolabel(rects): # useful to annotate bars with value
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
    """
        Plot horizontal 4-grouped bar plot to represent number of NaN values for each sensor sub-feature.
    """
    n = np.arange(0, len(feat), 1.1) # 10 feat
    h = 0.25

    # 4 bars group
    plt.barh(n + 0.05, values[np.arange(0, 40, 4)], color='xkcd:azure', height=h, label='mean')
    plt.barh(n + 0.25, values[np.arange(1, 40, 4)], color='xkcd:light red', height=h, label='min')
    plt.barh(n + 0.50, values[np.arange(2, 40, 4)], color='xkcd:medium green', height=h, label='max')
    plt.barh(n + 0.75, values[np.arange(3, 40, 4)], color='xkcd:dull pink', height=h, label='std')

    plt.title('NaN values for each feature')
    plt.xlabel('number of NaN values')
    plt.ylabel('sensors')

    # n = 0.25 * (len(sub feat) - 1) / 2
    n = 0.75/2 # first position

    plt.yticks(np.arange(n, 11, 1.1), feat)
    plt.legend()
    plt.show()
