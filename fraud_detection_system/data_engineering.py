"""_summary_

Raises:
    ValueError: _description_

Returns:
    _type_: _description_
"""

# standard python libraries
from datetime import datetime
from typing import Union, Dict
import re
# third party libraries
import pandas as pd
import numpy as np

class DataEngineering:
    """
    A class for performing various data engineering tasks including loading datasets,
    data cleaning, transformation, and validation.

    Attributes:
        file_name (str): The name of the file being processed.
        df (pd.DataFrame): The DataFrame loaded from the file.
    """
    def __init__(self, filename, **kwargs):
        """
        Initializes the DataEngineering class with a given file.

        Args:
            filename (str): The name of the file to load.
            **kwargs: Additional keyword arguments for data loading functions.
        """
        self.file_name = filename
        self.df = self.load_datasets(filename, **kwargs)

    def load_datasets(self, file_names: Union[list, str, pd.DataFrame], **kwargs) -> pd.DataFrame:
        """
        Loads datasets from a given file name, list of file names, or directly from a DataFrame.

        Args:
            file_names (Union[list, str, pd.DataFrame]): File name(s) or DataFrame to be loaded.
            **kwargs: Additional keyword arguments for reading functions.

        Raises:
            ValueError: If an unsupported file format is provided or if the input is not valid.

        Returns:
            pd.DataFrame: The loaded dataset as a DataFrame.
        """
        if isinstance(file_names, list):
            dataset = None
            for file_name in file_names:
                file_format = self._extract_format(file_name)
                if file_format not in ['csv', 'xlsx', 'parquet', 'pkl', 'json']:
                    raise ValueError("Unsupported file format")
                dataset = pd.concat([dataset, self.read_data(file_name, file_format, **kwargs)]) \
                    if dataset is not None else self.read_data(file_name, file_format, **kwargs)
            return dataset

        elif isinstance(file_names, str):
            file_format = self._extract_format(file_names)
            return self.read_data(file_names, file_format, **kwargs)

        elif isinstance(file_names, pd.DataFrame):
            return file_names

        else:
            raise ValueError("file_names needs to be a list of file names, a single file name\
                             , or a pd.DataFrame")

    def _extract_format(self, file_name: str) -> str:
        """
        Extracts the file format from the file name.

        Args:
            file_name (str): The file name from which to extract the format.

        Returns:
            str: The extracted file format.
        """
        match = re.search(r'\.([^\.]+)$', file_name)
        if match:
            return match.group(1)
        raise ValueError("File format could not be determined")

    def read_data(self, file_name: str, file_format: str, **kwargs) -> pd.DataFrame:
        """
        Reads data from a file based on its format.

        Args:
            file_name (str): The file name to read.
            file_format (str): The format of the file to guide which reading function to use.
            **kwargs: Additional keyword arguments for reading functions.

        Returns:
            pd.DataFrame: The data read from the file.
        """
        read_function = {
            'csv': pd.read_csv,
            'xlsx': pd.read_excel,
            'parquet': pd.read_parquet,
            'pkl': pd.read_pickle,
            'json': pd.read_json
        }

        if file_format in read_function:
            return read_function[file_format](file_name, **kwargs)
        else:
            raise ValueError(f"Unsupported file format: {file_format}")

    def describe(self) -> Dict:
        """
        Generates a JSON-style dictionary describing statistics for each column in the dataset.

        Returns:
            Dict: A dictionary where each key is a column name and each value is another dictionary containing
                  descriptive statistics for that column.
        """
        description = {
            'files_included': self.file_name,
            'column names': list(self.df.columns),
            'date ranges': (str(pd.to_datetime(self.df['trans_date_trans_time']).min()), str(pd.to_datetime(self.df['trans_date_trans_time']).max()))
        }
        measures = {
            'total_rows': int(len(self.df)),
            'fraud_cases': int(self.df['is_fraud'].sum()),
            'non_fraud_cases': int((self.df['is_fraud'] == 0).sum())
        }
        return {'description': description, 'measures': measures}

    # task 2: Data Cleaning Pipeline

    def clean_missing_values(self, args= None):
        """
        to detect and handle missing values, either by filling them or removing entries. If no args are passed, then
        it will by default use 'backfill' on all columns.
        :param args: list of tuples to indicate the column_name, fill_value, and fill_method to be passed onto
        self.clean_column_missing_values
        :return:
        """
        if not isinstance(args, list):
            for col in self.df.columns:
                self.clean_column_missing_values(column_name=col, fill_method='backfill')
        else:
            for arg in args:
                column_name = arg[0]
                fill_value = arg[1]
                fill_method = arg[2]
                self.clean_column_missing_values(column_name, fill_value, fill_method)

    def clean_column_missing_values(self, column_name: str, fill_value=None, fill_method: str = None):
        """
        Detect and handle missing values, either by filling them or removing entries for a given column_name.
        The fill_value or fill_method indicate the value or method to use.
        :param column_name: str. Column name within
        :param fill_value: value used to fill the empty instances of column_name
        :param fill_method: str: 'remove' will remove all na instances of column_name within self.dataset. Otherwise
        use 'backfill', 'bfill', 'ffill' according to
        https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.fillna.html
        :return: None
        """

        if column_name not in self.df.columns:
            print(f"""{column_name} is not a column in {self.df}""")
        else:
            if fill_value is not None:
                self.df[column_name].fillna(value=fill_value, axis=1, inplace=True)
            elif fill_method is not None:
                if fill_method == 'remove':
                    self.df = self.df[~self.df[column_name].isna()]
                    # Reset index after dropping rows
                    self.df.reset_index(drop=True, inplace=True)
                elif fill_method in ['backfill', 'bfill', 'ffill']:
                    self.df[column_name].fillna(method=fill_method, axis=1, inplace=True)
            else:
                print(f""" Please provide a valid fill_value or fill_method, refer to the documentation""")

    def remove_duplicates(self, subset_by: str= None, keep='first'):
        """
        Remove duplicate entries to ensure data integrity in self.dataset
        :param subset_by: column label or sequence of labels, optional
        :param keep: str: 'first','last', False will drop all duplicates.
        :return:
        """
        self.df.drop_duplicates(subset=subset_by, keep=keep, inplace=True)

    def standardize_dates(self, column_name: str):
        """
        Standardize and correct inconsistencies in date formats (i.e., as datetime) in self.dataset.
        :param column_name: str
        :return: None
        """
        self.df[column_name] = pd.to_datetime(self.df[column_name], errors='coerce')

    def trim_spaces(self, column_name: str):
        """
        Trim spaces from strings in categorical columns for consistency in self.dataset.
        :param column_name: str
        :return:
        """
        self.df[column_name] = self.df[column_name].astype(str)
        self.df[column_name] = self.df[column_name].str.upper()
        self.df[column_name] = self.df[column_name].str.strip()
        self.df[column_name] = np.where(self.df[column_name].isin(['NONE','NAN']),
                                             None,
                                             self.df[column_name])

    def resolve_anomalous_dates(self, column_name):
        """
        to resolve anomalies in date and time data in self.dataset, such as correcting future dates or improbable times.
        :param column_name:
        :return:
        """
        self.standardize_dates(column_name)
        now = datetime.now()
        self.df[column_name] = self.df[column_name].apply(lambda x: x if x <= now else now)
        self.df.dropna(subset=[column_name], inplace=True)
        self.df.reset_index(drop=True, inplace=True)

    # task 3: Data Transformation
    def expand_dates(self, column_name: str) -> pd.DataFrame:
        """
        to return a DataFrame that appends 'day_of_week' and 'hour_of_day' columns to self.dataset derived from the
        transaction to facilitate time-based analysis.
        :param column_name: string
        :return:
        """
        # in case that the column is not yet datetime
        # use standardize_dates to make it into datetime
        if not pd.api.types.is_datetime64_any_dtype(self.df[column_name]):
            self.standardize_dates(column_name)
        # add day_of_week
        self.df[f'{column_name}_day_of_week'] = self.df[column_name].dt.dayofweek

        # add is it a weekend
        self.df[f'{column_name}_weekend'] = np.where(self.df[f'{column_name}_day_of_week'].isna(), 
                                                          None, 
                                                          self.df[f'{column_name}_day_of_week'] >= 5)

        # add time_of_day
        # Define conditions and choices for categorizing time of day
        time = self.df[column_name].dt.hour
        conditions = [
            (time >= 5) & (time < 8),
            (time >= 8) & (time < 12),
            (time >= 12) & (time < 15),
            (time >= 15) & (time < 17),
            (time >= 17) & (time < 21),
            (time >= 21) | (time < 5),
            time.isna()
        ]
        choices = ['Early Morning', 'Morning', 'Afternoon', 'Late Afternoon','Evening', 'Late Evening', np.nan]

        # Use numpy.select to apply conditions and choices
        self.df[f'{column_name}_time_of_day'] = np.select(conditions, choices, default=np.nan)

        # add month
        self.df[f'{column_name}_month'] = self.df[column_name].dt.month

    def categorize_transactions(self):
        """
        that appends a column 'amt_category' to self.dataset that categorizes transaction amounts into "low"
        (bottom 25%), "medium" (25% to 75%), and "high" (above 75%) based on quantile ranges.
        :return: None
        """
        df = self.df.copy()
        quantiles = self.df['amt'].quantile([0.25, 0.75])
        low_threshold = quantiles[0.25]
        high_threshold = quantiles[0.75]

        # Function to categorize amounts
        def categorize(amount):
            if amount <= low_threshold:
                return 'low'
            elif amount <= high_threshold:
                return 'medium'
            else:
                return 'high'

        # Apply categorization
        self.df['amt_category'] = self.df['amt'].apply(categorize)

    # Task 4: Data Validation
    def range_checks(self, column_ranges: dict) -> pd.DataFrame:
        """
        Checks if the values in the numerical columns fall within the specified bounds.
        :param column_ranges: A dictionary where keys are column names and values are tuples (min, max)
                              specifying the expected range for each column.
        :return: list of out_of_bounds columns
        """
        out_of_bounds = list()
        for column, (min_val, max_val) in column_ranges.items():
            if column in self.df.columns:
                if ~self.df[column].between(min_val, max_val):
                    out_of_bounds.append(column)
        return out_of_bounds

    def null_checks(self, essential_columns: list) -> list:
        """
        Checks if there are any null values in the essential columns.
        :param essential_columns: A list of column names that are essential and should not have null values.
        :return: A list of essential columns that contain null values.
        """
        columns_with_nulls = []
        for column in essential_columns:
            if column in self.df.columns and self.df[column].isnull().any():
                columns_with_nulls.append(column)
        return columns_with_nulls

    def type_validation(self, expected_types: dict) -> list:
        """
        Validates that the data types of columns are consistent with expected formats.
        :param expected_types: A dictionary where keys are column names and values are expected data types.
        :return: A list of columns with data type inconsistencies.
        """

        inconsistent_columns = []
        for column, expected_type in expected_types.items():
            if column in self.df.columns:
                actual_type = self.df[column].dtype
                if not pd.api.types.is_dtype_equal(actual_type, expected_type):
                    inconsistent_columns.append(column)
        return inconsistent_columns

    def uniqueness_validation(self, unique_columns: list) -> pd.DataFrame:
        """
        Validates that the specified columns have unique values and checks for duplicate rows.
        :param unique_columns: A list of column names that should be unique for each transaction.
        :return: DataFrame containing duplicated rows based on the specified columns.
        """
        # Check for duplicate rows based on the specified unique columns
        duplicated_rows = self.df[self.df.duplicated(subset=unique_columns, keep=False)]
        if duplicated_rows.shape[0] > 0:
            return duplicated_rows
        else:
            return None

    def historical_data_consistency(self, new_data: pd.DataFrame, column_name: str,
                                    threshold: float = 3.0) -> pd.DataFrame:
        """
        Checks if new data entries are consistent with historical trends in the specified column.
        :param new_data: pd.DataFrane: A DataFrame containing new data entries to be checked.
        :param column_name: str: The name of the column to check for consistency.
        :param threshold: Float: The number of standard deviations to use as a threshold for flagging inconsistencies.
        :return: pd.Dataframe: DataFrame containing rows from new_data that are inconsistent with historical data.
        """

        # Calculate historical mean and standard deviation
        historical_mean = self.df[column_name].mean()
        historical_std = self.df[column_name].std()

        # Calculate the lower and upper bounds for consistency
        lower_bound = historical_mean - threshold * historical_std
        upper_bound = historical_mean + threshold * historical_std

        # Identify new data entries that are outside the bounds
        inconsistent_data = new_data[(new_data[column_name] < lower_bound) | (new_data[column_name] > upper_bound)]

        if inconsistent_data.shape[0] > 0:
            return inconsistent_data
        else:
            return None

    def categorical_data_validation(self, column_valid_categories: dict) -> pd.DataFrame:
        """
        Verifies that all entries in categorical fields match an approved list of categories.
        :param column_valid_categories: dict: A dictionary where keys are column names and values are lists of
        valid categories.
        :return: DataFrame containing rows with invalid categorical entries.
        """
        invalid_rows = pd.DataFrame()
        for column, valid_categories in column_valid_categories.items():
            if column in self.df.columns:
                invalid_entries = self.df[~self.df[column].isin(valid_categories)]
                invalid_rows = pd.concat([invalid_rows, invalid_entries])
        invalid_rows = invalid_rows.drop_duplicates().reset_index(drop=True)
        return invalid_rows
    
    @staticmethod
    def haversine_vectorized(lat1, lon1, lat2, lon2):
        """
        Calculate the great circle distance in kilometers between two points 
        on the earth specified in decimal degrees using a vectorized Haversine formula.
        
        Parameters:
        - lat1, lon1, lat2, lon2 : Arrays of latitudes and longitudes (decimal degrees)
        
        Returns:
        - Array of distances between each pair of points in kilometers.
        """
        # Convert all latitudes and longitudes from decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat / 2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0)**2
        c = 2 * np.arcsin(np.sqrt(a))

        # Radius of the Earth in kilometers
        r = 3956.0
        return c * r
    
    @staticmethod
    def distance_category(distance_series):    
        cond = [distance_series.isna(),
                distance_series>=80,
                distance_series>=40,
                distance_series>=20,
                distance_series<20]
        groups = [None,
                '>80',
                '40-80',
                '20-40',
                '<20']
        return np.select(cond, groups)
    
    @staticmethod
    def age_category(yob_series):    
        # we will define age-groups based on generations, for more information please refer to
        # https://libguides.usc.edu/busdem/age
        cond = [yob_series>=2013,
                yob_series>=1995,
                yob_series>=1980,
                yob_series>=1965,
                yob_series>=1946,
                yob_series>=1925,
                yob_series>=1901,
                yob_series.isna()]
        groups = ['Alpha',
                'Z',
                'Millennial',
                'X',
                'Baby Boomer',
                'Silent',
                'Greatest',
                None]
        return np.select(cond, groups)
