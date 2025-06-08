import torch

data_group = "Activities + Main sleep" # One of the keys in 'selected_features_list' below
special_mode_suffix = "" # Either "" or "_TL_Wearable_Korean"

debug_mode = False # if True, will not log to mlflow

naps = ['deep_count', 'deep_minutes', 'deep_thirtyDayAvgMinutes', 'light_count', 'light_minutes', 'light_thirtyDayAvgMinutes','rem_count', 'rem_minutes', 'rem_thirtyDayAvgMinutes','wake_count', 'wake_minutes', 'wake_thirtyDayAvgMinutes']
main_sleep = ['asleep_count', 'asleep_minutes', 'awake_count', 'awake_minutes','restless_minutes', 'restless_count']
activity = [
    "activityCalories", "caloriesBMR", "caloriesOut", "elevation",
    "fairlyActiveMinutes", "floors", "lightlyActiveMinutes", "marginalCalories",
    "restingHeartRate", "sedentaryMinutes", "steps", "veryActiveMinutes", "total",
    "activity_tracker", "activity_loggedActivities", "activity_veryActive",
    "activity_moderatelyActive", "activity_lightlyActive", "activity_sedentaryActive"
]
heart = [
    "heartRateZone_Out_ofRange_caloriesOut", "heartRateZone_Out_ofRange_max",
    "heartRateZone_Out_ofRange_min", "heartRateZone_Out_ofRange_minutes",
    "heartRateZone_Fat_Burn_caloriesOut", "heartRateZone_Fat_Burn_max",
    "heartRateZone_Fat_Burn_min", "heartRateZone_Fat_Burn_minutes",
    "heartRateZone_Cardio_caloriesOut", "heartRateZone_Cardio_max",
    "heartRateZone_Cardio_min", "heartRateZone_Cardio_minutes",
    "heartRateZone_Peak_caloriesOut", "heartRateZone_Peak_max",
    "heartRateZone_Peak_min", "heartRateZone_Peak_minutes"
]
common = ['age', 'race', 'gender']

selected_features_list = {
    "Activities": activity,
    "Activities + Heart rate": activity + heart,
    "Main sleep": main_sleep,
    "Naps": naps,
    "Activities + Main sleep": activity + naps,
    "Activities + Heart rate + Main sleep": activity + heart + main_sleep,
    "Activities + Heart rate + Naps": activity + heart + naps
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
num_epochs = 50
no_of_days = 5
num_layers = 3
metric_to_choose_best_model = 'val_auc'
hidden_size = 64
num_time_features = 8
prediction_length = 4
dropped_cols = []
selected_features = selected_features_list[data_group]
num_features = len(selected_features)
seq_length = no_of_days
val_split = 0.2
input_size = num_features
k_folds = 5
dropout = 0.5
freeze_threshold = 0.5