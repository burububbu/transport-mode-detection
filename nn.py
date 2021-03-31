import torch
import itertools
import numpy as np

from torch import nn
from torch.utils.data import Dataset, DataLoader

import visualization as vis

torch.backends.cudnn.benchmark = False

# class that represent dataset
class NNDataset(Dataset):
    """ Neural network model"""
    def __init__(self, x, y):
        self.num_classes = len(np.unique(y))
        self.X = torch.FloatTensor(x)
        self.y = torch.LongTensor(y)
    
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, ind):
        return self.X[ind, :], self.y[ind]

# then the instances will be passed to dataloader 
class NeuralNet(nn.Module):
    """ Neural network model"""
    def __init__(self, in_size, hidden_size, out_size, dropout):
        super(NeuralNet, self).__init__()

        self.input_size = in_size
        self.hidden_size = hidden_size
        self.output_size = out_size

        self.model = nn.Sequential(
                nn.Linear(in_size, hidden_size), #h1
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), #h2
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), #h3
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                
                nn.Dropout(dropout),
                nn.Linear(hidden_size, hidden_size), #h4
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size),
                
                nn.Linear(hidden_size, out_size), #out
            )

    def forward(self, x):
        out = self.model(x)
        return out

# def train loop
def _train_loop(dataloader, model, loss_fn, optimizer, device):
    model.train()

    loss_values = [] # all losses for each minibatch
    avg_loss = 0 # avg loss 

    for data, targets in dataloader:
        data = data.to(device)
        targets = targets.to(device)

        # Compute prediction and loss
        pred = model(data)
        loss = loss_fn(pred, targets)
        loss_values.append(loss)
        avg_loss = avg_loss + loss

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # model, all loss values for minibatch, avg loss
    return model, loss_values, avg_loss/len(dataloader)

# def test loop
def _test_loop(dataloader, model, loss_fn, device):
    model.eval()

    test_loss, correct = 0, 0 
    with torch.no_grad():
        for data, targets in dataloader:
            data = data.to(device)
            targets = targets.to(device)

            pred = model(data)

            test_loss = test_loss + loss_fn(pred, targets).item()
            correct = correct + (pred.argmax(1) == targets).type(torch.float).sum().item()

    test_loss = test_loss / len(dataloader) # divide test loss with number of samples

    model.train()

    # numero di valori per cui ci ha preso / il tot di valori
    return correct/len(dataloader.dataset), test_loss


def get_nn(x_tr, x_te, y_tr, y_te, hidden_s=20, epochs=100, batch_s=68, dropout=0.1, v=False, title=''):
    """
        If v == True then search for best hyperparameters for the model, use the given parameters otherwise.

        Returns: (NN instance, accuracy on test set), if v == True the instance is the best one
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("\t\tUsing {} device".format(device))

    torch.manual_seed(42)
    np.random.seed(42)
    torch.set_deterministic(True)

    # create a Dataset object for training data and another for test data
    training_data = NNDataset(x_tr.to_numpy(), y_tr)
    test_data = NNDataset(x_te.to_numpy(), y_te)

    if v:
        # hyperparameters space
        hidden_sizes = [25, 32, 50]
        nums_epochs = [200, 300, 500]
        batch_sizes = [16, 32, 68]
        dropout_sizes = [0, 0.2]

        # generate all possibile combinations
        hyperparameters = itertools.product(hidden_sizes, nums_epochs, batch_sizes, dropout_sizes)

        print('\t\tStarting hyperparameters tuning...')
        res = (None, 0)
        for hidden_size, num_epochs, batch_size, dropout_size in hyperparameters:

            print('\t\t\tWith hidden-size {}, epochs {}, batch size {}, dropout {}'.format(hidden_size, num_epochs, batch_size, dropout))
            new_res = _create_model(training_data, test_data, hidden_size, num_epochs, batch_size, dropout_size, device, title)
            
            if new_res[1] > res[1]:  # if the new model score is grater than the last, keep the new model 
                res = new_res
            
        return res
    else: # use the given parameters
        model, score= _create_model(training_data, test_data, hidden_s, epochs, batch_s, dropout, device, title)
        return model, score
        

def _create_model(training_data, test_data, hidden_s, epochs, batch_s, dropout, device, title):
    """
        Create, train and test a NeuralNet instance.
        
        Return: (NN instance, accuracy on test set)
    """
    model = NeuralNet(training_data.X.shape[1], hidden_s, training_data.num_classes, dropout)
    model.to(device)

    train_loader = DataLoader(training_data, batch_size=batch_s, shuffle = True)
    val_loader = DataLoader(test_data, batch_size=batch_s, shuffle = True)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)

    train_losses = [] # list of avg loss for each epoch on train set
    val_losses = [] # list of avg loss for each epoch on val set

    for e in np.arange(epochs):
        model, _, train_loss = _train_loop(train_loader, model, loss_fn, optimizer, device)
        _, val_loss = _test_loop(val_loader, model, loss_fn, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        scheduler.step()

        if (e+1)%50 == 0:
            print('\t\tepoch {}'.format(e+1))
            print('\t\t\tloss: {}'.format(train_loss))
    
    new_title = title + 'with hs={}, bs={}, d={}'.format(hidden_s, batch_s, dropout)
    fname = title+ '_{}_{}_{}_{}'.format(hidden_s, epochs, batch_s, dropout)

    vis.plot_loss(fname, new_title, train_losses, val_losses)

    # we have the trained model, now compute the accuracy
    train_score, _ = _test_loop(train_loader, model, loss_fn, device)
    test_score, _ = _test_loop(val_loader, model, loss_fn, device)

    print('\t\t\ttrain set score:{}, test set score:{}'.format(train_score, test_score))
    
    return model, test_score

    
