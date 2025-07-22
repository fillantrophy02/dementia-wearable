from enum import Enum
import os

import numpy as np
from sklearn.discriminant_analysis import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from config import *
import pandas as pd
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import GroupKFold, KFold, train_test_split


class DataframeLoader():
    def __init__(self):
        self.df = self._load_df()

    def _load_df(self):
        df = pd.read_csv(data_path)
        return df
    
    def get_df(self):
        return self.df
    
    @classmethod
    def split_df_into_sequences_with_labels(cls, df) -> tuple:
        df = df.sort_values(by=['participant', 'date'])
        feature_cols = [col for col in df.columns if col in selected_features and col not in (['date', 'label'])]
        feature_cols.insert(0, 'participant')      
        num_features = len(feature_cols)
        all_x = np.empty((0, seq_length, num_features))
        all_y = np.empty((0, 1))

        for _, group in df.groupby('participant'):
            group = group.set_index('date')  # Set date as index
            values = group[feature_cols + ['label']].to_numpy()

            if len(values) >= (seq_length):
                all_data = np.lib.stride_tricks.sliding_window_view(values, seq_length, axis=0)  # (N, num_features, seq_length+target_seq-1)
                all_data = np.transpose(all_data, (0, 2, 1)) # (N, seq_length+target_seq-1, num_features)
                x_data = all_data[:, 0:seq_length, :-1] # (N, seq_length, num_features)
                y_data = all_data[:, seq_length-1, -1:] # (N, target_seq_length, 1)

                all_x = np.concatenate((all_x, x_data), axis=0)
                all_y = np.concatenate((all_y, y_data), axis=0)

        return all_x, all_y

class SleepDataset(Dataset):
    def __init__(self, x, y = None, split = "train", activate_undersampling: bool = True, enable_normalization: bool = True, scaler: MinMaxScaler = None):
        self.split = split
        self.scaler = scaler
        self.x = x
        self.y = y

        if enable_normalization:
            #self._standardize()
            self._normalize()

        if activate_undersampling:
            self._undersample()

    def _standardize(self):
        """Applies Z-score standardization to self.x while maintaining its original shape."""
        num_samples, seq_len, num_features = self.x.shape  # Extract dimensions
        reshaped_x = self.x.reshape(-1, num_features)  # Flatten sequence dimension

        if isinstance(self.scaler, StandardScaler) and hasattr(self.scaler, "mean_"):
            standardized_x = self.scaler.transform(reshaped_x)  # Transform test data
        else:
            self.scaler = StandardScaler()
            standardized_x = self.scaler.fit_transform(reshaped_x)  # Fit and transform training data

        self.x = standardized_x.reshape(num_samples, seq_len, num_features)  # Reshape back to original form

    def _normalize(self):
        # min-max scaling
        num_samples, seq_len, num_features = self.x.shape  # Get shape of feature matrix
        reshaped_x = self.x.reshape(-1, num_features)

        if self.scaler:
            scaled_x = self.scaler.transform(reshaped_x)  # Only transform test data
        else:
            self.scaler = MinMaxScaler()
            scaled_x = self.scaler.fit_transform(reshaped_x)  # Fit and transform on train data
        self.x = scaled_x.reshape(num_samples, seq_len, num_features)   

    def _undersample(self):
        """Reduce majority class (label 0) to match the count of the minority class (label 1)."""
        indices_label_0 = np.where(self.y[:, 0, :] == 0)[0]
        indices_label_1 = np.where(self.y[:, 0, :] == 1)[0]

        num_label_1 = len(indices_label_1)
        sampled_indices_0 = np.random.choice(indices_label_0, size=num_label_1, replace=False)
        balanced_indices = np.concatenate([sampled_indices_0, indices_label_1])
        
        self.x = self.x[balanced_indices]
        self.y = self.y[balanced_indices]

    def get_samples_weight(self):
        y_flat = self.y.flatten().astype(int)
        num_label_0 = (y_flat == 0).sum() # only consider 'today' for label
        num_label_1 = (y_flat == 1).sum()
        class_sample_count = np.array([num_label_0, num_label_1])
        weight = 1. / class_sample_count
        samples_weight = np.array([weight[t] for t in y_flat])
        samples_weight = torch.from_numpy(samples_weight)
        return samples_weight

    def report(self):
        num_label_0 = (self.y[:, :] == 0).sum() # only consider 'today' for label
        num_label_1 = (self.y[:, :] == 1).sum()
        print(f"\n------ Stats for {self.split} --------")
        print(f"Number of samples with label 0: {num_label_0}")
        print(f"Number of samples with label 1: {num_label_1}")

    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):
        xs = torch.tensor(self.x[index], dtype=torch.float32)

        if self.y is None:
            return xs
        
        y = torch.tensor(self.y[index], dtype=torch.long)
        return xs, y
    

kf = GroupKFold(n_splits=k_folds, shuffle=True, random_state=40)
df = DataframeLoader().get_df()

train_dataloaders = []
val_dataloaders = []
val_ids = []
val_num_days = [] 
val_labels = []

for fold, (train_idx, val_idx) in enumerate(kf.split(df, groups=df['participant'])):
    print(f"\nFold {fold+1}/{k_folds}")

    # Split df by index
    df_train = df.iloc[train_idx]
    df_val = df.iloc[val_idx]

    # Split into sequences
    x_train, y_train = DataframeLoader.split_df_into_sequences_with_labels(df_train)
    x_val, y_val = DataframeLoader.split_df_into_sequences_with_labels(df_val)

    # Store and remove participant id
    x_train = x_train[:, :, 1:]
    val_id = x_val[:, -1, 0]
    x_val = x_val[:, :, 1:]

    # Create dataset
    train_ds = SleepDataset(x_train, y_train, "train", activate_undersampling=False, scaler=None)
    val_ds = SleepDataset(x_val, y_val, "val", activate_undersampling=False, scaler=train_ds.scaler)
    print(f"Train: {len(train_ds)} samples")
    print(f"Val: {len(val_ds)} samples")

    # Weighted sampler for training
    torch.manual_seed(24)
    samples_weight = train_ds.get_samples_weight()
    sampler = WeightedRandomSampler(samples_weight, len(samples_weight))

    # Create dataloaders
    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    # Append to lists
    train_dataloaders.append(train_loader)
    val_dataloaders.append(val_loader)

    # Operations for majorty vote counting
    val_num_day = {int(pid): list(val_id).count(pid) for pid in set(val_id)}
    val_label = {int(pid): label[0] for pid, label in zip(val_id, y_val)}
    val_ids.append(val_id)
    val_num_days.append(val_num_day)
    val_labels.append(val_label)

    train_ds.report()
    val_ds.report()

sample_train_features_batch, sample_train_labels_batch = next(iter(train_loader))
print(f"\nFeature batch shape: {sample_train_features_batch.size()}")
print(f"Labels batch shape: {sample_train_labels_batch.size()}")
