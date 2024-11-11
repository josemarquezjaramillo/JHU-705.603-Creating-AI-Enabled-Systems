import pickle
from feature_engineering import FeatureConstructor

class Model:
    """
    Model class that implement model behaviors including 
    versioning, training, and making predictions.
    """

    def __init__(self, model_name='random_forest_classifier'):
        """
        Initialize the Model with a specified architecture.

        """        
        self.model_name = model_name
        self.model_path = f"models/{model_name}.sav"
        self.model = self.set_model(self.model_path)

    def set_model(self, model_path):
        """
        Load a pre-trained model from a specified path.
        Parameters:
        model: 
        """        
        loaded_model = pickle.load(open(model_path, 'rb'))
        return loaded_model

    def train(self, dataset):
        """
        Train the model with provided training data.

        This function first processes the dataset using a feature construction process, which transforms the raw data
        into a format suitable for training the model (e.g., extracting features, scaling, etc.). After processing,
        it fits the model using the transformed features and targets.

        Args:
            dataset (DataFrame or similar): The raw dataset to train the model on. This dataset must be compatible with
                                           the `FeatureConstructor` class, which should handle necessary preprocessing.

        Raises:
            ValueError: If the dataset is empty or in an incorrect format that `FeatureConstructor` cannot handle.

        Notes:
            - The `FeatureConstructor` class is assumed to split the dataset into features (x) and target (y).
            - The actual implementation of `FeatureConstructor` should provide methods like `transform` to handle
              these operations.

        Example:
            >>> trainer = Model(model)
            >>> trainer.train(raw_dataset)
        """
        # Apply feature engineering to the dataset
        dataset = FeatureConstructor(dataset)  # Assumes existence of a class for feature transformation
        dataset.transform()  # Transform the dataset to feature matrix and target vector
        # Fit the model to the processed data
        self.model.fit(dataset.x, dataset.y)  # x are the features, y is the target

    def predict(self, dataset):
        """
        Make predictions based on processed input data.

        This method first applies necessary preprocessing to the raw input dataset using the FeatureConstructor
        class, which transforms the data into a feature matrix suitable for making predictions. The method
        then uses the pre-trained model to predict outcomes based on these features.

        Parameters:
            dataset (DataFrame or similar): The raw input data for prediction. This data must be preprocessed
                                            to transform into a format that the model can interpret, typically
                                            a numerical feature matrix.

        Returns:
            ndarray: An array of predictions made by the model. The type and shape of the returned array
                     depend on the model and the prediction task (e.g., classification or regression).

        Example:
            >>> my_model = Model(mode_name)
            >>> prediction_results = my_model.predict(new_data)
        """
        # Apply feature engineering to the input data
        feature_constructor = FeatureConstructor(dataset)
        feature_constructor.transform()  # Transform the data to a feature matrix suitable for the model

        # Use the trained model to make predictions on the processed features
        predictions = self.model.predict(feature_constructor.x)  # Return the predictions from the model

        return predictions
    
    def save_model(self, file_name):
        """
        Save the current model to a file using Python's pickle library.

        This method serializes the model object and writes it to a file, allowing the model to be
        loaded and used later without needing to retrain.

        Args:
            file_name (str): The path where the model should be saved. This should include the
                             file name and extension (typically .pkl).
        """
        # Open the file in binary write mode and save the pickle dump of the model
        with open(file_name, 'wb') as file:
            pickle.dump(self.model, file)  # Serialize the model object and write it to file
        