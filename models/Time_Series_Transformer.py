import math
import torch
import torch.nn as nn
import torchmetrics
from config import *
from transformers import TimeSeriesTransformerConfig, TimeSeriesTransformerModel

class TimeSeriesTransformer(nn.Module):
    '''Expect input of size (N, L, H_in) where N = batch_size, L = length of sequence, H_in = input_size aka no. of features'''
    def __init__(self):
        super().__init__()

        lag_length = 1
        lags_sequence = list(range(1, lag_length+1))

        config = TimeSeriesTransformerConfig(
            prediction_length=1,
            context_length=no_of_days - lag_length - 1,
            input_size=input_size,
            lags_sequence=lags_sequence,
            num_time_features=8,
            d_model=hidden_size,
            dropout=0.5
        )

        self.backbone = TimeSeriesTransformerModel(config).to(device)
        self.hidden1 = nn.Linear(hidden_size, 64)
        self.fc = nn.Linear(64, 1)
        self.dropout = nn.Dropout()
        
        self.activation = nn.ELU()
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.BCELoss()

    def forward(self, x):
        # sequence_length = no_of_days + max(lags_sequence)
        past_values, future_values = x[:, :1, :], x[:, 1:, :]
        past_mask = torch.ones_like(past_values).to(device)
        past_time_features = self.create_time_feature(past_values) 

        future_time_features = self.create_time_feature(future_values) 
        
        outputs = self.backbone(
            past_values=x,
            past_time_features=past_time_features,
            past_observed_mask=past_mask,
            future_values=future_values,
            future_time_features=future_time_features,
            output_hidden_states=True
        )
        h = outputs.last_hidden_state[:, -1:, :]
        h = torch.squeeze(h, 1) # (N, hidden_size)
        x = self.dropout(self.activation(self.hidden1(h)))
        x = self.sigmoid(self.fc(x))
        return x

    def create_time_feature(self, x): 
        # Sinusoidal encoding
        batch_size, sequence_length, _ = x.size()
        pos = torch.arange(sequence_length, device=device).unsqueeze(1)
        div = torch.exp(torch.arange(0, 8, 2, device=device) * (-math.log(10000.0) / 8))
        pe = torch.zeros(sequence_length, 8, device=device)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        return pe.unsqueeze(0).expand(batch_size, -1, -1)