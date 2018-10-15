import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

use_cuda = torch.cuda.is_available()

class TorchDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return len(self.x)

class TorchDataLoader:
    def __init__(self,batch_size,shuffle = True):
        self.batch_size = batch_size
        self.shuffle = shuffle
    
    def torch_dataloader(self,train_data,target_data):
        torch_dataset = TorchDataset(train_data,target_data)
        torch_loader = DataLoader(dataset = torch_dataset,
                                batch_size = self.batch_size, 
                                shuffle = self.shuffle)
        return torch_loader

def plot_results(predicted_data, true_data):
    # use in train.py 
    # plot evaluate result
    fig = plt.figure(facecolor='white')
    ax = fig.add_subplot(111)
    ax.plot(true_data, label='True Data')
    plt.plot(predicted_data, label='Prediction')
    plt.legend()
    plt.show()

def ToVariable(x):
    # use in train.py 
    # change from numpy.array to torch.variable   
    tmp = torch.DoubleTensor(x)
    if use_cuda:
        return Variable(tmp).cuda()
    else:
        return Variable(tmp)


