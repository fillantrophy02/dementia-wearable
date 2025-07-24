
import torch
import data.fitbit_mci.feature_list as fitbit_mci_features
import data.wearable_korean.feature_list as wearable_korean_features

dataset = "wearable_korean" # Either "fitbit_mci" or "wearable_korean"
data_group = "Sleep + Activity" # One of the keys in "<dataset>_features"
chosen_model = "PatchTST" # Either "PatchTST", "LSTM", "GRU", "VanillaTransformer" or "TimeSeriesTransformer"
is_transfer_learning = False # Only applies if "model" is set to "PatchTST"
freeze_threshold = 0.2 # Will unfreeze backbone after 20% of epochs
transfer_learning_dataset = "fitbit_mci" # Only applies if "is_transfer_learning" is set to True, either "wearable_korean" or "fitbit_mci"

if dataset == "fitbit_mci":
    selected_features = fitbit_mci_features.selected_features_list[data_group]
elif dataset == "wearable_korean":
    selected_features = wearable_korean_features.selected_features_list[data_group]

data_path = f"data/{dataset}/processed-data/dataset.csv"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
num_epochs = 40
no_of_days = 5
num_layers = 3
metric_to_choose_best_model = 'val_loss' # Either "val_loss" or "val_auc"
hidden_size = 64
prediction_length = 4
dropped_cols = []
num_features = len(selected_features)
seq_length = no_of_days
val_split = 0.2
input_size = num_features
k_folds = 5
dropout = 0.5
ffn_dim = 128
patch_length = 2
patch_stride = 1