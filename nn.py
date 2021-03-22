from torch.nn.modules.activation import ReLU
from torch.nn.modules.linear import Linear
from cons import BATCH_SIZE, PATH, TEST_SIZE, TO_EXCLUDE
import preprocessing as pre
from TDDataset import TDDataset
import torch
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from torch import nn
import numpy as np
from sklearn.preprocessing import LabelEncoder


class TMDataSet(Dataset):
    def __init__(self, x, y): # pass y as label encoded output
        self.num_classes = len(np.unique(y))
        self.X = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i) :
        return self.X[i, :], self.y[i]

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes

        # due layer, uno di input e uno hidden
        self.model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Linear(self.hidden_size, self.num_classes),
         )

    def forward(self, x):
        out = self.model(x)
        return out

def train_loop(dataloader, model, loss_fn, optimizer, epochs, device):
    #tell the model that you are training it 
    model.train()
    loss_values = []
    for e in range(epochs):    
        for data, targets in dataloader:
            print("{} epoch".format(e))
            data, targets = data.to(device), targets.to(device)

            # forward pass
            pred = model(data)
            
            # compute loss
            loss = loss_fn(pred, targets)
            loss_values.append(loss)

            # backpropagation
            loss.backward()

            optimizer.step()
    return loss_values
        

def test_loop(model, dataloader, device):
    model.eval() # stato di valutazione
    score = 0

    with torch.no_grad(): # vale solo all'interno dello statement
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
            pred = model(data) #forward
            sm = torch.nn.Softmax()
            print('{}'.format(sm(pred)))

            correct = (pred.argmax(1) == targets).type(torch.float).sum().item() 
            score = score + correct
    score = score/ len(dataloader.dataset)
    print({'Accuracy: {}'.format(score*100)})  

 
def main():
    hidden_size = 200
    # initial hyperparameters settings

    learning_rate = 1e-3
    batch_size = 64
    epochs = 100

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} device".format(device))

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_deterministic(False)

    # loaded dataset and created 
    dt = TDDataset(PATH, TO_EXCLUDE)
    le = LabelEncoder()
    le.fit(dt.data['target'])
    # 1a. preprocessing
    dt.remove_sensor_feat('step_counter')
    dt.split_train_test(TEST_SIZE, prep = True) # fill Nan with median, drop duplicates

    x_tr, x_te, y_tr, y_te = dt.get_train_test_sensors(dt.feat)

    x_tr_st, x_te_st = pre.standardization(x_tr, x_te)
    print(x_tr.shape)

    # passare questi 
    y_tr_enc = le.transform(y_tr) 
    y_te_enc = le.transform(y_te)

    # quindi alla fine passare
    # x_tr_st, y_tr_enc
    # x_te_st, y_te_enc

    training_data = TMDataSet(x_tr_st.to_numpy(), y_tr_enc)
    test_data = TMDataSet(x_te_st.to_numpy(), y_te_enc)

    train_dataloader = DataLoader(training_data, batch_size= batch_size) 
    test_dataloader = DataLoader(test_data, batch_size= batch_size) 

    #  training_data.X.shape[1] -> numero di features, di colonne
    model = NeuralNetwork(training_data.X.shape[1], hidden_size, training_data.num_classes).to(device)

    print(model)

    # now start the optimization loop = {the train loop, the validation/test loop}
    # initialize loss function
    loss_fn = nn.CrossEntropyLoss()
    # utilize stochastic gradient descend
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_loop(train_dataloader, model, loss_fn, optimizer, epochs, device)
    test_loop(model, test_dataloader, device)





if __name__ == '__main__':
    main()