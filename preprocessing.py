# FILE DIVISO I DUE PARTI
# 1. analisi del dataset
#       controllare se dataset è bilanciato
#       controllare quandi sample ci sono (e quanti sono nan)
#       (relatici grafici)

import matplotlib.pyplot as plt
import utils

def check_balanced_dt(dt):
    # remember -> group by without Nan
    temp = dt.groupby(['target']).count()
    pos = 1
    for i in range(0,40,4):
        plt.subplot(2, 5, pos)

        plt.title(utils.clean_name(temp.iloc[:, i].name))
        plt.bar(temp.index, temp.iloc[:, i])

        
        pos = pos + 1

    plt.show()
    # TODO inoltre voglio sapere, per ogni feature (#mean), quanti sample non sono nan



# 2. preprocessing
#       effettuare preprocessing personalizzato in abse al tipo di model

