import pandas as pd
import json
from typing import Union, List, Dict, Tuple
import datetime

from data_engineering import DataEngineering


class DatasetConstructor(DataEngineering):
    """
    A class to construct and manage datasets for fraud detection analysis.

    Attributes:
        dataset (pd.DataFrame): The main dataset used for operations.
        args (tuple): Additional positional arguments.
        kwargs (dict): Additional keyword arguments.
        data_sources (list): List of data sources used to construct the dataset.
        required_keys (dict): Contains keys required for operations.
    """

    def __init__(self, dataset: DataEngineering, output_filename, format='parquet', sample_size=10000):
        """
        Initializes the DatasetConstructor with raw data and optional parameters.

        Args:
            raw_data (Union[str, pd.DataFrame]): The path to the dataset file or a DataFrame.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        self.dataset = dataset
        self.data_sources = self.dataset.file_name
        self.version = self.set_version()
        self.new_dataset = self.sample(n=sample_size)        
        self.log = self.describe(dataframe=self.new_dataset)
        self.dataset.standardize_dates('trans_date_trans_time')
        self.dataset.standardize_dates('dob')
        self.load(output_filename, format)

    def load(self, output_filename: str = None, format: str = None) -> None:
        """
        Save the dataset to a file in a specified format.

        Args:
            output_filename (str): The filename to which the dataset will be saved.
            format (str, optional): The format in which to save the dataset. Defaults to None.
        """
        output_filename = output_filename

        if format == 'csv':
            self.new_dataset.to_csv(output_filename + ".csv", index=False)
        elif format == 'parquet':
            self.new_dataset.to_parquet(output_filename + ".parquet", coerce_timestamps='ms', allow_truncated_timestamps=True)
        elif format == 'json':
            self.new_dataset.to_json(output_filename + ".json", orient='records')
        else:
            raise ValueError("Unsupported file format specified.")

    def get_data_source(self) -> List[str]:
        """Return the list of data sources."""
        return self.data_sources

    def set_version(self) -> str:
        """Set and return a unique version identifier based on the current date and time."""
        return f"{datetime.datetime.now().isoformat()}"

    def sample(self, n) -> pd.DataFrame:
        """
        Implements stratified sampling based on the is_fraud proportion on the dataset loaded

        Args:
            n: number of samples to be returned

        Returns:
            pd.DataFrame: A sampled subset of the dataset.
        """
        
        samples = self.dataset.df['is_fraud'].value_counts()/self.dataset.df['is_fraud'].count()*n
        samples = samples.round().astype(int).to_dict()
        df = None
        for x in samples:
            if df is None:
                df = self.dataset.df[self.dataset.df['is_fraud']==x].sample(n=samples[x])
            else:
                df = pd.concat([df, self.dataset.df[self.dataset.df['is_fraud']==x].sample(n=samples[x])])
        return df

    def describe(self, dataframe: pd.DataFrame) -> Dict:
        """
        Generate a description of the dataset including various statistics and measures.

        Args:
            dataframe (pd.DataFrame): The DataFrame to describe.
            output_file (str, optional): Path to save the description JSON file.

        Returns:
            Dict: A dictionary containing descriptive statistics and data measures.
        """
        description = {
            'version': self.version,
            'data sources': self.get_data_source(),
            'column names': list(dataframe.columns),
            'date ranges': (str(pd.to_datetime(dataframe['trans_date_trans_time']).min()), str(pd.to_datetime(dataframe['trans_date_trans_time']).max()))
        }
        measures = {
            'total_rows': int(len(dataframe)),
            'fraud_cases': int(dataframe['is_fraud'].sum()),
            'non_fraud_cases': int((dataframe['is_fraud'] == 0).sum())
        }
        output = {'description': description, 'measures': measures}
        #if output_file:
        #    with open(output_file + ".json", 'w') as file:
        #        json.dump(output, file, indent=4, default=str)
        return output


# Example usage:
if __name__ == "__main__":
    import os
    os.chdir('fraud_detection_system/')
    data_path = ['data/transactions_0.csv','data/transactions_1.parquet']  # Modify this path as needed
    dataset = DataEngineering(data_path)
    constructor = DatasetConstructor(dataset, output_filename='dataseta')
    #constructor.load('datasetA','json')
    #sample_data = constructor.sample(100)
    #description = constructor.describe(constructor.dataset, 'fraud_detection_system/datasets/dataset_description')
    #print(description)
