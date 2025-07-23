import os
import pandas as pd
import matplotlib.pyplot as plt

dataset_fp = 'data/processed-data/dataset.csv'
output_folder = 'visualizations'
patient_no = 0

# Define features to plot
top_5_important_features = [
    'sleep_breath_average',
    'sleep_rmssd',
    'sleep_rem',
    'sleep_deep',
    'sleep_restless'
]

df = pd.read_csv(dataset_fp)

# Ensure only the first patient's data is selected
first_patient_email = df['EMAIL'].unique()[patient_no]
patient_df = df[df['EMAIL'] == first_patient_email]
output_fp = os.path.join(output_folder, f'time_series_patient_{patient_no}.png')

# Convert date to datetime
patient_df['date'] = pd.to_datetime(patient_df['date'])

# Define a colormap
colors = ['orange', 'blue', 'green', 'red', 'purple']

# Create a single figure with multiple subplots
fig, axes = plt.subplots(len(top_5_important_features), 1, figsize=(20, 15), sharex=True)

for ax, feature, color in zip(axes, top_5_important_features, colors):
    if feature in patient_df.columns:
        ax.plot(patient_df['date'], patient_df[feature], marker='o', linestyle='-', color=color)
        ax.set_ylabel(feature, color=color)
        ax.grid()
        ax.set_xticks(patient_df['date'])  # Ensure all dates are shown
        ax.set_xticklabels(patient_df['date'].dt.strftime('%Y-%m-%d'), rotation=45)
    else:
        print(f"Warning: {feature} not found in dataset.")

# Formatting the figure
plt.xlabel('Date')
plt.xticks(rotation=45)
plt.suptitle('Time Series of Selected Sleep Features')
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig(output_fp)

print(patient_df[:5])