import torch

special_mode_suffix = "_TL_Wearable_Korean" # Either "" or "_TL_Wearable_Korean"

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 256
num_epochs = 10
no_of_days = 5
num_layers = 3
metric_to_choose_best_model = 'val_auc'
hidden_size = 64
num_time_features = 8
prediction_length = 4
dropped_cols = []
excluded_features = activity + heart + main_sleep
num_features = 67 - len([x for x in excluded_features if x not in dropped_cols])
seq_length = no_of_days
val_split = 0.2
input_size = num_features
k_folds = 5
dropout = 0.5
freeze_threshold = 0.5