from data import fitbit_mci
import data.wearable_korean.feature_list as wearable_korean_features

data_group = "Korean-Fitbit Common Features" # One of the keys in '<dataset>.selected_features_list' 
selected_features = wearable_korean_features.selected_features_list[data_group]
batch_size = 256
num_epochs = 350
no_of_days = 5
num_layers = 3
metric_to_choose_best_model = 'val_loss'
num_features = len(selected_features)
prediction_length = 4
cutoff = no_of_days - prediction_length
hidden_size = 64
nhead = 4
input_size = num_features # no of features
dropout = 0.8
freeze_threshold = 0.1
ffn_dim = 512
patch_length = 2
patch_stride = 1