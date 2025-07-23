import torch
import torch.nn as nn
import torchmetrics
from config import *

class VanillaTransformer(nn.Module):
    '''Expect input of size (N, L, H_in) where N = batch_size, L = length of sequence, H_in = input_size aka no. of features'''
    def __init__(self):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=input_size, nhead=4, dropout=0.5, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.hidden1 = nn.Linear(input_size, 64)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout()
        
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        x = self.encoder(x) # (N, L, H_in)
        x = x[:, -1, :]
        x = self.dropout(self.activation(self.hidden1(x)))
        x = self.sigmoid(self.fc(x))
        return x