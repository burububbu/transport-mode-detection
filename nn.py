import itertools
import torch

import numpy as np
import pandas as pd
from torch import nn
from torch.utils.data import Dataset, DataLoader

torch.backends.cudnn.benchmark = False

# then the instances will be passed to dataloader 
class NNDataset(Dataset):
    def __init__(self, x, y):
        self.num_classes = len(np.unique(y))
        self.X = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, ind):
        return self.X[ind, :], self.y[ind]


class NeuralNet(nn.Module):
    """ Neural network model"""
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(NeuralNet, self).__init__()

        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = out_size

        # self.model = nn.Sequential(
        #     nn.Dropout(dropout),
        #     nn.Linear(in_size, hidden_size), #h1
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, hidden_size), #h2
        #     nn.ReLU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, hidden_size), #h3
        #     nn.ReLU(), 
        #     nn.Linear(hidden_size, out_size), #out
        # )

        self.model = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_size, hidden_size),  #h1
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),  #h2
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),  #h3
            nn.ReLU(),
            nn.Linear(hidden_size, out_size),  #out
        )

    def forward(self, x):
        out = self.model(x)
        return out

def train_loop(dataloader, model, loss_fn, optimizer, epochs, device, valloader = None):
    """
        For each epoch iterate over dataset searching for optimal results.
        If valloader != None then for each epoch will be tested the model. 
    
        Return: (model, list of all loss values, list of loss values means for epoch)
    """
    model.train()

    n_batch = len(dataloader)

    loss_mean_epochs = []
    loss_values = []

    print('Evaluating ...')
    for e in range(epochs):
        acc = 0
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            # forward pass
            pred = model(data)

            # loss computation
            loss = loss_fn(pred, targets)
            loss_values.append(loss)
            acc = acc + loss.item()

            # backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        loss_mean_epochs.append(acc/n_batch) 

        print(' epoch {}'.format(e+1))
        print('\tloss: {}'.format(acc/n_batch))

        if valloader:
            print('\t test loss: {}'.format(test_loop(valloader, model, loss_fn, device)[1]))
            model.train()

    return model, loss_values, loss_mean_epochs


def test_loop(dataloader, model, loss_fn, device):
    """
        Test model, iterate over the val/test dataset

        Return (accuracy, mean test loss)  
    """
    model.eval()

    test_loss, correct = 0, 0

    with torch.no_grad():
        for data, targets in dataloader:
            data, targets = data.to(device), targets.to(device)

            pred = model(data)

            test_loss = test_loss + loss_fn(pred, targets).item()
            correct = correct + (pred.argmax(1) == targets).type(torch.float).sum().item() # correct predictions

    # average of minibatches loss values  
    test_loss = test_loss/len(dataloader)

    return correct/len(dataloader.dataset), test_loss


def get_nn(x_tr, x_te, y_tr, y_te, v=False, hs=10, lr=0.01, epochs=100, bs=64, drop=0.2):
    """
        Train mlp. If v = True then search for best hyperparameters else train with given parameters.

        Return: (model, score) if v = False, (results df, best score) otherwise
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using {} device".format(device))

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_deterministic(True)

    # create one Dataset object for training set and another for test
    training_data = NNDataset(x_tr.to_numpy(), y_tr)
    test_data = NNDataset(x_te.to_numpy(), y_te)

    val_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    if v: # search for best hyperparams
        results = pd.DataFrame(columns=['params', 'train_acc', 'test_acc'])

        # hyperparameters space
        hidden_sizes = [20, 50, 100]
        nums_epochs = [50, 100]
        batch_sizes = [8, 16, 32]
        learning_rate = [0.1, 0.01]
        dropout = [0.2, 0.5]

        hyperparameters = itertools.product(hidden_sizes, nums_epochs, batch_sizes, learning_rate, dropout)

        print('Starting hyperparameters tuning...')
        for hidden_size, l_rate, epochs, batch_size, dropout in hyperparameters:
            
            train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)

            model = NeuralNet(training_data.X.shape[1], hidden_size, training_data.num_classes, dropout)
            model.to(device)

            loss_fn = nn.CrossEntropyLoss()
            optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)

            # train the model 
            model, _, means = train_loop(train_loader, model, loss_fn, optimizer, epochs, device)
            # plt.plot(mean)

            score, _ = test_loop(val_loader, model, loss_fn, device)
            train_acc = test_loop(train_loader, model, loss_fn, device)[0]
            
            print('With hidden-size {}, learning_rate {}, epochs {}, batch size {}, dropout {}'.format(hidden_size, l_rate, epochs, batch_size, dropout))
            print('Train set accuracy: {}'.format(train_acc))
            print('Val set accuracy: {}'.format(score))

            # append information to results dataframe
            results.append(
                [[{'hs': hidden_size, 'lr': l_rate, 'ep': epochs, 'b':batch_size, 'dr': dropout},
                    train_acc,
                    score]
                    ])

            return results, results['test_acc'].max()

    else:  # here there isn't hyperparams tuning
        print('With hidden-size {}, learning_rate {}, epochs {}, batch size {}'.format(hs, lr, epochs, bs)) 
        train_loader = DataLoader(training_data, batch_size= bs, shuffle=True)

        model = NeuralNet(training_data.X.shape[1], hs, training_data.num_classes, drop)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)\

        model, _, means = train_loop(train_loader, model, loss_fn, optimizer, epochs, device, valloader=val_loader)
        score, _ = test_loop(val_loader, model, loss_fn, device)
        #plt.plot(means)
        print('Val set accuracy: {}'.format(score))
        
        return model, score


def temp(training_data, val_loader, hidden_size, l_rate, epochs, batch_size, dropout, device, v = False):
        train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)       
    
        model = NeuralNet(training_data.X.shape[1], hidden_size, training_data.num_classes, dropout)
        model.to(device)

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=l_rate)

        # train the model 
        model, _, means = train_loop(train_loader, model, loss_fn, optimizer, epochs, device)
        # plt.plot(mean)

        val_score, val_avg_loss  = test_loop(val_loader, model, loss_fn, device)
        train_score, train_avg_loss = test_loop(train_loader, model, loss_fn, device)[0]
    
        print('With hidden-size {}, learning_rate {}, epochs {}, batch size {}, dropout {}'.format(hidden_size, l_rate, epochs, batch_size, dropout))
        print('Train set accuracy: {}, Avg loss: {}'.format(train_score, train_avg_loss))
        print('Val set accuracy: {}, Avg loss: {}'.format(val_score, val_avg_loss))

        # return information to append to the final results in hyperparameters search
        if v:
            return [
                {'hs': hidden_size, 'lr': l_rate, 'ep': epochs, 'b':batch_size, 'dr': dropout},
                train_score,
                val_score
                ]
        
        
        
        # results.append(
        #     [
        #         ])