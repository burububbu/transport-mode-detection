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
C = {'D1': 600,'D2': 400,'D3': 400,}

KNN_GRID = {'n_neighbors': [2, 4, 6, 8, 10]}
N_NEIGHBORS = {'D1': 4,'D2': 2,'D3': 2,}

# best NN params
HIDDEN_SIZE = {'D1': 25,'D2': 50,'D3': 32}
EPOCHS = {'D1': 300,'D2': 200,'D3': 200}
BATCH_SIZE = {'D1': 16,'D2': 32,'D3': 68}
DROPOUT = {'D1': 0,'D2': 0.1,'D3': 0.2}

