import torch

debug_mode = False # if True, will not log to mlflow

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
num_epochs = 300
no_of_days = 5
num_layers = 3
metric_to_choose_best_model = 'val_auc'
hidden_size = 64
num_time_features = 8
prediction_length = 4
excluded_features = []
num_features = 25 - len(excluded_features)
seq_length = no_of_days
val_split = 0.2