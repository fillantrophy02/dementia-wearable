naps =  ['minutesAfterWakeup', 'minutesAsleep', 'minutesAwake', 'minutesToFallAsleep', 'timeInBed', 'totalMinutesAsleep', 'totalSleepRecords', 'totalTimeInBed', 'asleep_count', 'asleep_minutes', 'restless_count', 'restless_minutes', 'awake_count', 'awake_minutes']
main_sleep = ['minutesAfterWakeup', 'minutesAsleep', 'minutesAwake', 'minutesToFallAsleep', 'timeInBed', 'totalMinutesAsleep', 'totalSleepRecords', 'totalTimeInBed', 'deep_count', 'deep_minutes', 'deep_thirtyDayAvgMinutes', 'light_count', 'light_minutes', 'light_thirtyDayAvgMinutes', 'rem_count', 'rem_minutes', 'rem_thirtyDayAvgMinutes', 'wake_count', 'wake_minutes', 'wake_thirtyDayAvgMinutes']
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
korean_fitbit_common_features = [
    'deep_minutes',
    'light_minutes',
    'asleep_minutes',
    'activityCalories',
    'caloriesOut',
    'fairlyActiveMinutes',
    'lightlyActiveMinutes',
    'sedentaryMinutes',
    'steps',
    'veryActiveMinutes'
]

selected_features_list = {
    "Activities": activity,
    "Activities + Heart rate": activity + heart,
    "Main sleep": main_sleep,
    "Naps": naps,
    "Activities + Main sleep": activity + main_sleep,
    "Activities + Naps": activity + naps,
    "Activities + Heart rate + Main sleep": activity + heart + main_sleep,
    "Activities + Heart rate + Naps": activity + heart + naps,
    "Korean-Fitbit Common Features": korean_fitbit_common_features
}