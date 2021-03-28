PATH = '../dataset/dataset_5secondWindow.csv'
TO_EXCLUDE = ['light', 'pressure', 'magnetic_field','gravity','proximity']

PLOT_PATH = './plots/'
TEST_SIZE = 0.30

# True to activate hyperparameters tuning
TO_VAL = True
NN_TO_VAL = False

# model params
RF_ESTIMATORS = 200

SVM_GRID = {'C': [200, 400, 600]}
C = {'D1': 600,'D4': 400,'D5': 400,}

KNN_GRID = {'n_neighbors': [2, 4, 6, 8, 10]}
N_NEIGHBORS = {'D1': 4,'D4': 2,'D5': 2,}

# best params neural networ
# HIDDEN_SIZE = {'D1': 50,'D4': 100,'D5': 300}
# L_RATE ={'D1': 0.01,'D4': 0.1,'D5': 0.1}
# EPOCHS = {'D1': 200,'D4': 200,'D5': 200}
# BATCH_SIZE = {'D1': 32,'D4': 16,'D5': 32}
# DROPOUT = {'D1': 0.1,'D4': 0.5,'D5': 0.5}

HIDDEN_SIZE = {'D1': 25,'D4': 50,'D5': 32} # BEST 4 DEEP NN
L_RATE ={'D1': 0.1,'D4': 0.1,'D5': 0.1}
EPOCHS = {'D1': 300,'D4': 200,'D5': 199} # D5: 200
BATCH_SIZE = {'D1': 16,'D4': 32,'D5': 68}
DROPOUT = {'D1': 0,'D4': 0,'D5': 0.2}
DECAY = {'D1': True,'D4': False,'D5': True}

