
import torch
import data.fitbit_mci.feature_list as fitbit_mci_features

dataset = 'fitbit_mci'
is_transfer_learning = True
transfer_learning_dataset = "wearable_korean"

data_group = "Korean-Fitbit Common Features" # One of the keys in '<dataset>.selected_features_list' 
chosen_model = "PatchTST" # Either "PatchTST" or "LSTM"

if dataset == "fitbit_mci":
    selected_features = fitbit_mci_features.selected_features_list[data_group]
data_path = f"data/{dataset}/processed-data/dataset.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
num_epochs = 30
no_of_days = 5
num_layers = 3
metric_to_choose_best_model = 'val_loss'
hidden_size = 64
prediction_length = 4
dropped_cols = []
num_features = len(selected_features)
seq_length = no_of_days
val_split = 0.2
input_size = num_features
k_folds = 5
dropout = 0.5
freeze_threshold = 0.5
ffn_dim = 512
patch_length = 2
patch_stride = 1