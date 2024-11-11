from feature_engineering import FeatureConstructor
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

class Metrics:
    """
    Metrics class for evaluating binary classification models with additional evaluation metrics.
    """

    def __init__(self, dataset, model):
        """
        Initialize the Metrics class with a model and dataset. It preprocesses the dataset and predicts labels.
        
        Args:
            dataset (DataFrame or similar): The raw dataset to be used for model evaluation.
            model (Model): The machine learning model to be evaluated.
        """
        self.model = model
        self.dataset = FeatureConstructor(dataset)
        self.dataset.transform()
        self.predicted_labels = self.model.model.predict(self.dataset.x)

    def accuracy(self):
        """
        Calculate the accuracy of the model.
        
        Returns:
            float: The accuracy score.
        """
        return accuracy_score(self.dataset.y, self.predicted_labels)

    def precision(self):
        """
        Calculate the precision of the model.
        
        Returns:
            float: The precision score.
        """
        return precision_score(self.dataset.y, self.predicted_labels)

    def recall(self):
        """
        Calculate the recall of the model.
        
        Returns:
            float: The recall score.
        """
        return recall_score(self.dataset.y, self.predicted_labels)

    def f1(self):
        """
        Calculate the F1 score of the model.
        
        Returns:
            float: The F1 score.
        """
        return f1_score(self.dataset.y, self.predicted_labels)

    def confusion_matrix(self):
        """
        Calculate the confusion matrix of the model predictions.

        Returns:
            ndarray: A confusion matrix of shape (2, 2), as a list of lists.
        """
        cm = confusion_matrix(self.dataset.y, self.predicted_labels)
        return cm.tolist()  # Convert numpy array to list for JSON serialization

    def false_positive_rate(self):
        """
        Calculate the false positive rate (FPR) of the model.
        
        Returns:
            float: The false positive rate.
        """
        cm = confusion_matrix(self.dataset.y, self.predicted_labels)
        TN, FP, FN, TP = cm.ravel()
        return FP / (FP + TN)

    def run_metrics(self):
        """
        Compute and return all metrics including the confusion matrix and FPR.
        
        Returns:
            dict: A dictionary containing accuracy, precision, recall, F1 score, confusion matrix, and FPR.
        """
        return {
            'accuracy': self.accuracy(),
            'precision': self.precision(),
            'recall': self.recall(),
            'f1_score': self.f1(),
            'confusion_matrix': self.confusion_matrix(),
            'false_positive_rate': self.false_positive_rate()
        }
