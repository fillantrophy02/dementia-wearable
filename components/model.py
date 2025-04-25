import math
import torch
import torch.nn as nn
import torchmetrics
from config import *
from transformers import PatchTSTConfig, PatchTSTModel

class SleepPatchTST(nn.Module):
    '''Expect input of size (N, L, H_in) where N = batch_size, L = length of sequence, H_in = input_size aka no. of features'''
    def __init__(self, input_size):
        super().__init__()

        config = PatchTSTConfig(
            num_input_channels=input_size,
            context_length=no_of_days,
            patch_length=2,
            patch_stride=1,
            num_hidden_layers=num_layers,
            d_model=hidden_size,
            attention_dropout=0.5,
            ff_dropout=0.5,
            scaling=None,
            prediction_length=prediction_length,
            num_targets=input_size
        )

        self.backbone = PatchTSTModel(config).to(device)
        self.hidden1 = nn.Linear(input_size * hidden_size, 64)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout()
        
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        # sequence_length = no_of_days + max(lags_sequence)
        cutoff = no_of_days - prediction_length
        past_values, future_values = x[:, :cutoff, :], x[:, cutoff:, :]
        past_mask = torch.ones_like(past_values).to(device)
        # past_time_features = self.create_time_feature(past_values) 

        outputs = self.backbone(
            past_values=x,
            past_observed_mask=past_mask,
            future_values=future_values,
        )
        h = outputs.last_hidden_state # (N, input_size, no_of_days, hidden_size)
        h = h[:, :, -1, :] # (N, input_size, hidden_size)
        h = torch.flatten(h, 1) # (N, input_size * hidden_size)
        x = self.dropout(self.activation(self.hidden1(h)))
        x = self.sigmoid(self.fc(x))
        return x

    def create_time_feature(self, x): 
        # Sinusoidal encoding
        batch_size, sequence_length, _ = x.size()
        pos = torch.arange(sequence_length, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, num_time_features, 2, device=device) * (-math.log(10000.0) / num_time_features))
        pe = torch.zeros(sequence_length, num_time_features, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0).expand(batch_size, -1, -1)