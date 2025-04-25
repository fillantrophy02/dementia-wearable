import pandas as pd
from data.data_processing import DataProcessing, DataType
import pytest

@pytest.fixture(scope="module")
def df_sleep():
    return DataProcessing().create_combined_dataframe(DataType.SLEEP)

def test_create_sleep_dataframe_has_correct_number_of_samples(df_sleep):
    expected_no_of_samples = 12183
    assert len(df_sleep) == expected_no_of_samples

def test_create_sleep_dataframe_has_correct_number_of_features(df_sleep):
    expected_no_of_features = 32

    no_of_features_excluding_email_date_and_label = len(df_sleep.columns) - 3
    assert no_of_features_excluding_email_date_and_label == expected_no_of_features

def test_create_sleep_dataframe_has_correct_number_of_patients(df_sleep):
    expected_no_of_patients = 174

    no_of_patients = len(df_sleep['EMAIL'].unique())
    assert no_of_patients == expected_no_of_patients

def test_create_sleep_dataframe_has_correct_number_of_normal_functioning_patients(df_sleep):
    expected_no_of_patients = 111

    no_of_patients = df_sleep[df_sleep['label'] == 0]['EMAIL'].nunique()
    assert no_of_patients == expected_no_of_patients

def test_create_sleep_dataframe_has_correct_number_of_cognitively_impaired_patients(df_sleep):
    expected_no_of_patients = 63

    no_of_patients = df_sleep[df_sleep['label'] == 1]['EMAIL'].nunique()
    assert no_of_patients == expected_no_of_patients

def test_create_sleep_dataframe_has_correct_record(df_sleep):
    email, date = "nia+354@rowan.kr", "2020-12-28"
    expected_record = {
        "sleep_awake": 8100,
        "start1": 0,
        "start2": 0,
        "start3": 0,
        "start4": 0,
        "start5": 0,
        "start6": 1,
        "end1": 0,
        "end2": 1,
        "end3": 0,
        "end4": 0,
        "end5": 0,
        "end6": 0,
        "sleep_breath_average": 16,
        "sleep_deep": 7890,
        "sleep_duration": 29940,
        "sleep_efficiency": 73,
        "sleep_hr_average": 53.02,
        "sleep_hr_lowest": 50,
        "sleep_light": 13830,
        "sleep_midpoint_at_delta": 9899,
        "sleep_midpoint_time": 13470,
        "sleep_onset_latency": 360,
        "sleep_rem": 120,
        "sleep_restless": 41,
        "sleep_rmssd": 45,
        "sleep_temperature_delta": -0.2,
        "sleep_temperature_deviation": -0.2,
        "sleep_total": 21840,
        "sleep_hr_max": 60,
        "sleep_hr_median": 52,
        "sleep_hypnogram_average": 2.26,
        "label": 0
    }

    record = df_sleep[df_sleep['EMAIL'] == email]
    record = record[pd.to_datetime(record['date']) == pd.to_datetime(date)]
    for feature in expected_record:
        value = record[feature].values[0]
        expected_value = expected_record[feature]
        assert value == expected_value, f"Error: Expected value {expected_value} for feature {feature}, but received value {value}"


