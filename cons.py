PATH = '../dataset/dataset_5secondWindow.csv'
TO_EXCLUDE = ['light', 'pressure', 'magnetic_field','gravity','proximity']

TEST_SIZE = 0.30

# True to activate hyperparameters tuning
TO_VAL = False
NN_TO_VAL = False

# model params
RF_ESTIMATORS = 200

SVM_GRID = {'C': [200, 400, 600]}
C = {'D1': 600,'D4': 400,'D5': 400,}

KNN_GRID = {'n_neighbors': [2, 4, 6, 8, 10]}
N_NEIGHBORS = {'D1': 4,'D4': 2,'D5': 2,}

# best params neural network
HIDDEN_SIZE = {'D1': 50,'D4': 50,'D5': 100}
L_RATE ={'D1': 0.01,'D4': 0.01,'D5': 0.01}
EPOCHS = {'D1': 200,'D4': 200,'D5': 200}
BATCH_SIZE = {'D1': 8,'D4': 16,'D5': 32}
DROPOUT = {'D1': 0.1,'D4': 0.5,'D5': 0.2}