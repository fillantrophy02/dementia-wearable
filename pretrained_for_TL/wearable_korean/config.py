from data import fitbit_mci
import data.fitbit_mci.feature_list as fitbit_mci_features

data_group = "Korean-Fitbit Common Features" # One of the keys in '<dataset>.selected_features_list' 
selected_features = fitbit_mci_features.selected_features_list[data_group]
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