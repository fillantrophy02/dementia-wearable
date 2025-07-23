

# Convert raw data to 'train.csv' and 'test.csv' that include 174 patients and 32 sleep features
# Current implemented for sleep features only

from enum import Enum
import statistics
import pandas as pd
from config import *

from data.wearable_korean.utils import categorize_sleep_end_hour, categorize_sleep_start_hour, get_average, get_max, get_median


class DataType(Enum):
    ACTIVITY = 'activity'
    MMSE = 'msse'
    SLEEP = 'sleep'

class DataSet(Enum):
    TRAIN = 'training'
    TEST = 'validation'

class DataFeature(Enum):
    INPUT = 'origin'
    OUTPUT = 'label'

label_dict = {'CN': 0, 'MCI': 1, 'Dem': 1}

class DataProcessing():
    def __init__(self):
        pass

    def create_csv_file(self):
        fp = data_path
        df = self._combine_different_data_types_dfs()
        df.to_csv(fp, index=False)
        return df
    
    def _combine_different_data_types_dfs(self):
        df_sleep = self._combine_train_and_test_dfs(DataType.SLEEP)
        df_activity = self._combine_train_and_test_dfs(DataType.ACTIVITY)
        df = pd.merge(df_sleep, df_activity, on=['EMAIL', 'date', 'label'], how='inner')
        df = df[[col for col in df if col != 'label'] + ['label']]
        print(f"Merged sleep & activity, ignoring {len(df_sleep) - len(df)} sleep and {len(df_activity) - len(df)} activity rows, forming {len(df)} rows and {len(df.columns)} columns.")
        return df

    def _combine_train_and_test_dfs(self, type: DataType) -> pd.DataFrame:
        df_train = self._create_dataframe(type, DataSet.TRAIN)
        df_test = self._create_dataframe(type, DataSet.TEST)
        df = pd.concat([df_train, df_test])
        return df
    
    def _create_dataframe(self, type: DataType, set: DataFeature):
        fp = self._build_filepath(type, set, DataFeature.INPUT)
        df = pd.read_csv(fp)
        df = self._generate_features(df, type)
        df = self._remove_irrelevant_features(df, type)
        df = self._concatenate_output_column(df, type, set)
        df = self._sort_columns(df)
        return df
    
    def _build_filepath(self, type: DataType, set: DataSet, feature: DataFeature):
        fp = 'data/raw-data' # base directory
        fp += f'/{set.value}'
        fp += f'/{feature.value}_data'
        fp += f'/{type.value}'
        fp += f'/{'train' if set == DataSet.TRAIN else 'val'}'
        fp += f'_{type.value}'
        fp += f'{'_label' if feature == DataFeature.OUTPUT else ''}'
        fp += f'.csv'
        return fp

    def _concatenate_output_column(self, df, type, set):
        fp_output = self._build_filepath(type, set, DataFeature.OUTPUT)
        df_output = pd.read_csv(fp_output)
        df = df.merge(df_output, left_on='EMAIL', right_on='SAMPLE_EMAIL', how='left')
        df['label'] = df['DIAG_NM'].map(label_dict)
        df = df.drop(columns=df_output.columns)
        return df

    def _generate_features(self, df: pd.DataFrame, type: DataType):
        '''See feature_selection.csv for more details'''
        if type == DataType.SLEEP:
            extra_stats_columns = pd.DataFrame({
                'sleep_hr_max': df['CONVERT(sleep_hr_5min USING utf8)'].apply(get_max),
                'sleep_hr_median': df['CONVERT(sleep_hr_5min USING utf8)'].apply(get_median),
                'sleep_hypnogram_average': df['CONVERT(sleep_hypnogram_5min USING utf8)'].apply(get_average)
            })

            df['date'] = pd.to_datetime(df['sleep_bedtime_end']).dt.date

            # If a person sleeps twice within the same day, shift it to the next day
            df['date_offset'] = df.groupby(['EMAIL', 'date']).cumcount()  # Group by email and date
            df['date'] = df['date'] + pd.to_timedelta(df['date_offset'], unit='D')  # Shift by days
            df.drop(columns=['date_offset'], inplace=True)  # Clean up

            sleep_start_hour = pd.to_datetime(df['sleep_bedtime_start']).dt.hour
            sleep_start_cols = sleep_start_hour.apply(categorize_sleep_start_hour).apply(pd.Series)

            sleep_end_hour = pd.to_datetime(df['sleep_bedtime_end']).dt.hour
            sleep_end_cols = sleep_end_hour.apply(categorize_sleep_end_hour).apply(pd.Series)

            df = pd.concat([df, extra_stats_columns, sleep_start_cols, sleep_end_cols], axis=1)
        elif type == DataType.ACTIVITY:
            df['date'] = pd.to_datetime(df['activity_day_end']).dt.date
            df['date_offset'] = df.groupby(['EMAIL', 'date']).cumcount()  # Group by email and date
            df['date'] = df['date'] + pd.to_timedelta(df['date_offset'], unit='D')  # Shift by days
            df.drop(columns=['date_offset'], inplace=True)  # Clean up
        return df   

    def _sort_columns(self, df: pd.DataFrame):
        priority = {'EMAIL': 0, 'date': 1, 'label': 99999}
        sorted_columns = sorted(df.columns, key=lambda x: priority.get(x, 2))
        return df[sorted_columns]

    def _remove_irrelevant_features(self, df: pd.DataFrame, type: DataType):
        id_column = 'EMAIL'
        feature_selection_df = pd.read_csv(f'data/feature_selection_{type.value}.csv')
        selected = feature_selection_df.loc[feature_selection_df["Chosen"] == 1, "Features"].to_list()
        selected.insert(0, id_column)
        return df[selected]
    
if __name__ == '__main__':
    DataProcessing().create_csv_file()