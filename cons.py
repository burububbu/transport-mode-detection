PATH = '../dataset/dataset_5secondWindow.csv'
TO_EXCLUDE = ['light', 'pressure', 'magnetic_field','gravity','proximity']

RF_ESTIMATORS = 200
KNN_GRID = {'n_neighbors': [2, 4, 6, 8, 10]}
SVM_GRID = {'C': [200, 400, 600]}
TEST_SIZE = 0.20

BATCH_SIZE = 64
