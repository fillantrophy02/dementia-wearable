import torch
import torch.nn as nn
import torchmetrics
from config import *

class GRU(nn.Module):
    '''Expect input of size (N, L, H_in) where N = batch_size, L = length of sequence, H_in = input_size aka no. of features'''
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden_size=hidden_size, num_layers=num_layers, dropout=0.5, batch_first=True)
        self.hidden1 = nn.Linear(num_layers * hidden_size, 64)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout()
        
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        _, h = self.gru(x) # (num_layers, N, hidden_size)
        h = torch.transpose(h, 0, 1) # (N, num_layers, hidden_size)
        h = torch.flatten(h, 1) # (N, hidden_size * num_layers)
        x = self.dropout(self.activation(self.hidden1(h)))
        x = self.sigmoid(self.fc(x))
        return x