from cons import PATH, TEST_SIZE, TO_EXCLUDE
import preprocessing as pre
from TDDataset import TDDataset

from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import itertools
import numpy as np

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, Subset

torch.backends.cudnn.benchmark = False

# class that represent dataset
class TMDataset(Dataset):
    def __init__(self, x, y):
        self.num_classes = len(np.unique(y))
        self.X = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, ind):
        return self.X[ind, :], self.y[ind]


class NeuralNet(nn.Module):
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(NeuralNet, self).__init__()
        
        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = out_size

        self.model = nn.Sequential(
            nn.Linear(in_size, hidden_size), #h1
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size), #h2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, out_size), #out
        )

    def forward(self, x):
        out = self.model(x)
        return out


def train_loop(dataloader, model, loss_fn, optimizer, epochs, device):
    # training mode
    model.train()
    
    n_batch = len(dataloader)
    loss_mean_epochs = []
    acc = 0
    print('Evaluating ...')
    loss_values = [] # tutte le loss per ogni batch
    for e in range(epochs):
        # print('Starting epoch {}'.format(e+1))
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            # forward pass + loss computation
            pred = model(data)

            loss = loss_fn(pred, targets)
            loss_values.append(loss)
            acc = acc + loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_mean_epochs.append(acc/n_batch) 
    # model, all loss values for ever batch (len(batches) * epoch), mean loss for every epoch
    return model, loss_values, loss_mean_epochs

def test_loop(dataloader, model, loss_fn, device):
    # evaluation mode
    model.eval()

    test_loss, correct = 0, 0
    
    size = len(dataloader.dataset)
    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)
           
            pred = model(data)

            test_loss  = test_loss +  loss_fn(pred, targets).item()
            correct = correct +  (pred.argmax(1) == targets).type(torch.float).sum().item()

    # media della loss di ogni batch (che forma il dataset)
    test_loss = test_loss/len(dataloader)
    # numero predizioni corrette sul totale del dataset
    correct = correct/size
    
    # accuracy, avg loss
    return correct*100, test_loss


if __name__ == '__main__':
    # setting the device to use
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} device".format(device))

    # set here hyperparameters
    # TODO set hyp (these are examples)
    # hidden_sizes = [10, 100, 300, 600] # numero neuroni del mio hidden layer
    # nums_epochs = [100, 500]
    # batch_sizes = [16, 32, 65]

    # 10, 100,
    hidden_sizes = [100, 300, 600] # numero neuroni del mio hidden layer
    nums_epochs = [200, 500]
    batch_sizes = [16, 32]
    learning_rate = [0.001, 0.01]
    
    hyperparameters = itertools.product(hidden_sizes, nums_epochs, batch_sizes, learning_rate)

    # load dataset
    dt = TDDataset(PATH, TO_EXCLUDE)

    le = LabelEncoder()
    le.fit(dt.data['target'])
    # 1a. preprocessing
    dt.remove_sensor_feat('step_counter')
    dt.split_train_test(TEST_SIZE, prep = True) # fill Nan with median, drop duplicates

    D1 = ['accelerometer','sound', 'orientation']   
    D4 = ['accelerometer','sound','orientation','linear_acceleration','gyroscope', 'rotation_vector']
    x_tr, x_te, y_tr, y_te = dt.get_train_test_sensors(D1)

    x_tr_st, x_te_st = pre.standardization(x_tr, x_te)
    
    # encording
    y_tr_enc = le.transform(y_tr) 
    y_te_enc = le.transform(y_te)
    
    training_data = TMDataset(x_tr_st.to_numpy(), y_tr_enc)
    test_data = TMDataset(x_te_st.to_numpy(), y_te_enc)

    val_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    # for della cross val
    for hidden_size, epochs, batch_size, l_rate in hyperparameters:
        torch.manual_seed(42)
        np.random.seed(42)
        torch.set_deterministic(True)
        print('With hidden-size {}, learning_rate {}, epochs {}, batch size {}'.format(hidden_size, l_rate, epochs, batch_size))
        
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

        model = NeuralNet(training_data.X.shape[1], hidden_size, training_data.num_classes, 0.2)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=l_rate)

        model, loss_values, _ = train_loop(train_loader, model, loss_fn, optimizer, epochs, device)

        score, _ = test_loop(val_loader, model, loss_fn, device)

        print('With hidden-size {}, learning_rate {}, epochs {}, batch size {}'.format(hidden_size, l_rate, epochs, batch_size)) 
        print('accuracy on train = {}'.format(test_loop(train_loader, model, loss_fn, device)[0]))
        print('accuracy on test = {}'.format(score))

    




