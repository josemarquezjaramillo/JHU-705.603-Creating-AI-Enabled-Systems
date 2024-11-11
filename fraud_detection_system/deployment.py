import json
import os

from data_engineering import DataEngineering
from dataset import DatasetConstructor
from model import Model
from metrics import Metrics

class DeploymentPipeline:
    """
    Manages the deployment processes for machine learning models, facilitating training,
    inference, and logging across various stages of the ML lifecycle.

    Attributes:
        dataset (DataEngineering): A DataEngineering instance to handle data transformations.
        model (Model): A Model instance for making predictions and training.
    """

    def __init__(self, raw_data, model_name='random_forest_classifier'):
        """
        Initializes the DeploymentPipeline with a dataset and model configuration.

        Args:
            raw_data (str or DataFrame): The path to the raw data file or a DataFrame.
            model (str, optional): Path to a pre-trained model file. Defaults to a random forest classifier.
        """
        # Initialize Data Engineering with raw data
        self.dataset = DataEngineering(raw_data)
        # Load the model from the specified file
        self.model = Model(model_name)
        self.metrics = Metrics(raw_data, self.model)

    def predict(self, data, model='random_forest_classifier'):
        """
        Predicts outputs based on the given input data using a specified model.

        Args:
            data (dict): The input data formatted as a dictionary for prediction.
            model (str): The model to use for prediction, allows switching models dynamically.

        Returns:
            dict: The prediction results from the model.
        """
        # Check if the specified model differs from the currently loaded model and update if necessary
        if model != self.model.model_name:
            self.model = Model(model)
        # Make a prediction with the current model
        prediction = self.model.predict(data)
        return prediction
    
    def train_model(self, data, model, new_model_name):
        """
        Trains the model with specified data.

        Args:
            data (DataFrame or dict): The training data.
            model (str): Path to the model that needs training or retraining.

        Returns:
            Model: The trained model object.
        """
        # Update model if a different one is specified
        if model != self.model.model_name:
            self.model = Model(model)
        # Train the model using provided data
        self.model.train(data)
        # save the trained model in the new_model_name
        self.model.save_model(f'models/{new_model_name}.sav')
        # load new model into the current model
        self.model = Model(new_model_name)
        # load the new model into the metrics        
        self.metrics.model = Model(new_model_name)


    def get_log(self, component, reference):
        """
        Fetches log entries for a specific component using a reference ID.

        Args:
            component (str): The component type (e.g., 'dataset', 'model').
            reference (str): The reference ID for retrieving the log.

        Returns:
            dict: The log details retrieved from the log file.

        Raises:
            FileNotFoundError: If the log file does not exist.
            ValueError: If there are issues decoding the log file.
        """
        log_path = f"resources/logs/{component}/{reference}.json"
        try:
            with open(log_path, "r") as logs:
                description = json.load(logs)
        except FileNotFoundError:
            raise FileNotFoundError(f"Log file not found: {log_path}")
        except json.JSONDecodeError:
            raise ValueError(f"Error decoding JSON from log file: {log_path}")

        return description

    def generate_new_dataset(self, output_filename, n=1000):
        """
        Generates a new dataset and logs the process.

        Args:
            output_filename (str): Filename to store the generated dataset.
            n (int, optional): Number of samples to generate.

        Returns:
            log: Details of the dataset generation process.
        """
        new_dataset = DatasetConstructor(self.dataset, output_filename, sample_size=n, format='csv')
        return new_dataset.log
    
    def ingest_new_data(self, datafile):
        """
        Ingests new data into the pipeline.

        Args:
            datafile (str): The path to the data file.

        Returns:
            log: A log object with details about the ingestion process.
        """
        ingested_data_json_log = DataEngineering(datafile)
        return ingested_data_json_log
        
    def log(self, component, log_entry, log_file):
        """
        Logs details about different components of the pipeline to specified files.

        Args:
            component (str): The component type ('model', 'dataset', etc.).
            log_entry (dict): The log entry to record.
            log_file (str): The filename for the log entry.
        """
        # Ensure the directory for the logs exists
        os.makedirs(f'resources/logs/{component}', exist_ok=True)
        # Write the log entry to the specified file
        with open(f'resources/logs/{component}/{log_file}', "w") as log:
            json.dump(log_entry, log, indent=4)
