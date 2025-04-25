# only implemented for 'sleep', might expand to 'activity' and 'heart' also
from functools import reduce
from enum import Enum
import statistics
import pandas as pd

class DataType(Enum):
    SLEEP = 'sleep'
    ACTIVITY = 'activity'

class DataProcessing():
    def __init__(self):
        pass

    def create_csv_file(self):
        fp = 'data/processed-data/dataset.csv'
        self._create_dataframe() 
        self.df.to_csv(fp, index=False)
        return self.df
    
    def _create_dataframe(self):
        dfs = []
        dfs.append(pd.read_csv(self._build_filepath(DataType.SLEEP)))
        dfs.append(pd.read_csv(self._build_filepath(DataType.ACTIVITY)))
        self.df = self._merge_dfs(dfs)
        self.df = self._sort_columns(self.df)
        self._drop_columns_with_too_many_missing_values() # 38 -> 28 columns
        self._interpolate_missing_values()
        return self.df
    
    def _build_filepath(self, type: DataType):
        fp = 'data/raw-data' # base directory
        fp += f'/{type.value}.csv'
        return fp
    
    def _merge_dfs(self, dfs):
        return reduce(lambda left, right: pd.merge(left, right, on=['participant', 'date'], how='inner'), dfs)

    def _sort_columns(self, df: pd.DataFrame):
        priority = {'participant': 0, 'date': 1, 'label': 99999}
        sorted_columns = sorted(df.columns, key=lambda x: priority.get(x, 2))
        return df[sorted_columns]
    
    def _drop_columns_with_too_many_missing_values(self, threshold=0.8):
        missing_fraction = self.df.isnull().mean() 
        columns_to_drop = missing_fraction[missing_fraction > threshold].index  
        self.df.drop(columns=columns_to_drop, inplace=True)
        print(f"Dropping columns: {columns_to_drop}")

    def _interpolate_missing_values(self):
        self.df.interpolate(inplace=True, limit_direction='both')

    def report(self):
        num_columns = self.df.shape[1]
        num_rows = self.df.shape[0]
        if 'participant' in self.df.columns:
            unique_locations = self.df['participant'].nunique()
        avg_rows_per_location = num_rows / unique_locations if unique_locations > 0 else 0
        num_empty_cells = self.df.isnull().sum().sum()
        
        report_str = (
            f"Number of columns: {num_columns}\n"
            f"Number of rows: {num_rows}\n"
            f"Number of missing values: {num_empty_cells}\n"
            f"Number of unique locations: {unique_locations}\n"
            f"Average number of rows per location: {avg_rows_per_location:.0f}\n"
        )
        
        print(report_str)
        # print(self.df.head())
        print(self.df.iloc[:20])
    
if __name__ == '__main__':
    obj = DataProcessing()
    obj.create_csv_file()
    obj.report()
