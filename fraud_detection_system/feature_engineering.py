# Python Standard Libraries
import json
from typing import List, Dict
import datetime

# Third Party Libraries
import pandas as pd
import numpy as np

# ML Libraries
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline  # To include SMOTE in the pipeline
from imblearn.under_sampling import RandomUnderSampler

# User defined modules
from data_engineering import DataEngineering

class FeatureConstructor(DataEngineering):
    """
    A class to construct and manage feature engineering processes for datasets used in fraud detection.

    Attributes:
        dataset (pd.DataFrame): The dataset being manipulated and transformed.
        data_sources (list): List of data sources from which the dataset is constructed.
    """

    def __init__(self, raw_data, **kwargs):        
        """
        Initializes the FeatureConstructor with raw data and optional parameters.

        Args:
            raw_data (Union[str, pd.DataFrame]): The path to the dataset file or a DataFrame.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        """
        DataEngineering.__init__(self, filename= raw_data, **kwargs)
        self.data_sources = raw_data
        self.version = self.set_version()
        self.x = None
        self.y = None
        self.feature_names = None

    def get_data_sources(self) -> List[str]:
        """
        Return the list of data sources used to construct the dataset.

        Returns:
            List[str]: A list of data source file names or descriptions.
        """
        return self.data_sources

    def set_version(self) -> str:
        """
        Set and return a unique version identifier based on the current date and time.

        Returns:
            str: A version identifier as a string.
        """
        return f"{datetime.datetime.now().isoformat()}"
    
    def transform(self) -> pd.DataFrame:
        """
        Apply transformations to the dataset to create new features or modify existing ones.

        Args:
            *args: Variable length argument list for transformation parameters.
            **kwargs: Arbitrary keyword arguments for transformation parameters.

        Returns:
            pd.DataFrame: The transformed dataset.
        """
        self.pre_process()
        self.x, self.y, self.feature_names = self.pipeline(smote_sampling=False)

    def pre_process(self) -> pd.DataFrame:
        """
        Implements cleaning and pre-processing steps
        """
        # Data Pre-Processing
        # cleaning missing values
        self.clean_missing_values([('is_fraud',None,'remove')])
        # remove duplicate transactions
        self.remove_duplicates(subset_by='trans_num')
        # resolve anomalous dates
        for col in ['trans_date_trans_time', 'dob']:
            self.standardize_dates(col)
        # expand dates on transaction dates
        self.expand_dates('trans_date_trans_time')
        # trim spaces
        for col in ['cc_num','merchant','category','first','last','sex','street','city','state','job']:
            self.trim_spaces(col)

        # Distance Features
        # compute haversine distance
        self.df['distance_mi'] = self.haversine_vectorized(self.df['lat'],
                                                                  self.df['long'],    
                                                                  self.df['merch_lat'], 
                                                                  self.df['merch_long'])       
        # define the distance category
        self.df['distance_cat'] = self.distance_category(self.df['distance_mi'])

        # Age Features
        self.df['age_years'] = np.where(self.df['dob'].isna(),
                                            None,
                                            (pd.Timestamp.today() - self.df['dob'])  / pd.Timedelta(365, 'D')
                                            )
        self.df['yob'] = self.df['dob'].dt.year
        self.df['age_group'] = self.age_category(self.df['yob'])
    
    def pipeline(self, smote_sampling=False):
        """
        Preprocesses the given dataset for fraud detection modeling.
        
        This function handles numerical and categorical data preprocessing and can optionally
        apply SMOTE for handling class imbalance in training datasets.
        
        Parameters:
            dataset (DataFrame): The input dataset containing features and a target labeled 'is_fraud'.
            use_smote (bool): Flag to determine whether to apply SMOTE for oversampling the minority class.

        Returns:
            Tuple (np.ndarray, pd.Series, np.ndarray):
                - Processed features after scaling and encoding.
                - Resampled target variable if SMOTE is used.
                - One-hot encoded target variable.
        """
        # Separating features and target variable
        x = self.df[[col for col in self.df.columns if col != 'is_fraud']]
        y = self.df['is_fraud']
        
        # Defining attributes for different data processing methods
        # Numeric features
        num_attributes = ['amt',
                          'age_years',
                          'distance_mi'] 
        # Categorical Features 
        cat_attributes = ['distance_cat',
                          #'sex',
                          'age_group',
                          'category',
                          'trans_date_trans_time_weekend',
                          #'trans_date_trans_time_day_of_week',
                          'trans_date_trans_time_time_of_day',
                          #'trans_date_trans_time_month'
                          ]

        # Categorical Value
        cat_values = [['40-80', '<20', '20-40', '>80',None],
                      #[None, 'F', 'M'],
                      ['Millennial', None, 'Silent', 'Baby Boomer', 'X', 'Z', 'Greatest'],
                      ['MISC_NET',  'GROCERY_POS',  'ENTERTAINMENT',  'GAS_TRANSPORT',  'MISC_POS',  'GROCERY_NET',  None,  'SHOPPING_POS',
                       'SHOPPING_NET',  'FOOD_DINING',  'PERSONAL_CARE',  'HEALTH_FITNESS',  'TRAVEL',  'KIDS_PETS',  'HOME'],
                      [True, False],
                      #[1, 2, 3, 4, 5, 6, 0],
                      ['Late Evening',  'Early Morning',  'Morning',  'Afternoon',  'Late Afternoon',  'Evening'],
                      #[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                      ]

        names = [[str(cat_attributes[i])+'_'+str(cat_values[i][j]) for j in range(len(cat_values[i]))] for i, cat in enumerate(cat_attributes)]
        feature_names = num_attributes.copy()
        for name in names:
            feature_names.extend(name)  

        std_scaler =  StandardScaler()
        # Based on the training Data implement values for scaling
        std_scaler.mean_ = [70.306486,50.794746,47.254624]
        std_scaler.scale_ = [159.053100,17.412408,18.072797]

        # Creating pipelines for numeric and categorical feature processing
        num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median', fill_value=0, keep_empty_features=True)),  # Imputing missing values with median
            ('std_scaler', std_scaler)                # Standardizing features
        ])

        cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent', fill_value='N/A', keep_empty_features=True)),  # Imputing missing values with mode
        ('one_hot', OneHotEncoder(categories=cat_values))                           # Applying one-hot encoding
            ])
        
        # Combining numeric and categorical pipelines into a column transformer
        full_pipeline = ColumnTransformer([
            ("num", num_pipeline, num_attributes),
            ("cat", cat_pipeline, cat_attributes)
        ])

        # Optional SMOTE inclusion based on use_smote flag
        if smote_sampling:
            # Include SMOTE in the pipeline if use_smote is True
            pipeline = ImbPipeline([
                ('preprocessor', full_pipeline),
                ('smote_over', SMOTE(random_state=42, sampling_strategy=0.05)),
                ('random_under', RandomUnderSampler(sampling_strategy=0.1))  # SMOTE for oversampling
            ])
        else:
            # Regular pipeline without SMOTE
            pipeline = Pipeline([
                ('preprocessor', full_pipeline)
            ])
        
        # Applying the pipeline to process features and optionally apply SMOTE
        if smote_sampling:
            x_processed, y_resampled = pipeline.fit_resample(x, y)
        else:
            x_processed = pipeline.fit_transform(x)
            y_resampled = y

        return x_processed, y_resampled, feature_names

    def describe(self, dataframe: pd.DataFrame, output_file: str = None) -> Dict:
        """
        Generate a description of the dataset including various statistics and measures related to the features.

        Args:
            dataframe (pd.DataFrame): The DataFrame to describe.
            output_file (str, optional): Path to save the description JSON file.

        Returns:
            Dict: A dictionary containing descriptive statistics and data measures.
        """
        description = {
            'version': self.version,
            'data sources': self.get_data_sources(),
            'column names': list(dataframe.columns),
            'date ranges': (dataframe['trans_date_trans_time'].min(), dataframe['trans_date_trans_time'].max())
        }
        measures = {
            'total_rows': len(dataframe),
            'fraud_cases': int(dataframe['is_fraud'].count()),
            'non_fraud_cases': int((dataframe['is_fraud'] == 0).count())
        }
        output = {'description': description, 'measures': measures}
        if output_file:
            with open(output_file + ".json", 'w') as file:
                json.dump(output, file, indent=4, default=str)
        return output


# Example usage:
if __name__ == "__main__":
    data_path = 'fraud_detection_system/data/transactions_0.csv'  # Modify this path as needed
    constructor = FeatureConstructor(data_path)
