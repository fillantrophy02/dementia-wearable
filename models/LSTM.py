import math
import torch
import torch.nn as nn
import torchmetrics
from config import *

class LSTM(nn.Module):
    '''Expect input of size (N, L, H_in) where N = batch_size, L = length of sequence, H_in = input_size aka no. of features'''
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size, 128, 1, batch_first=True, dropout=0.5)
        self.hidden1 = nn.Linear(128, hidden_size)
        self.fc = nn.Linear(hidden_size, 1)
        
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()
        self.auc = torchmetrics.AUROC(task='binary')
        self.cm = torchmetrics.ConfusionMatrix(task='binary')
        self.f1_score = torchmetrics.F1Score(task='binary')

    def forward(self, x):
        x, (h, c) = self.lstm(x)
        x = x[:, -1, :] # take only last time step
        x = self.activation(self.hidden1(x))
        x = self.sigmoid(self.fc(x))
        return x
