import numpy as np
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt

class Metrics:

    def __init__(self):
        pass

    def calculate_ious(self, boxes):
        """
        Compute the Intersection over Union (IoU) of each pair of bounding boxes using vectorized operations.

        Parameters:
        - boxes: NumPy array of bounding boxes with shape (N, 4), where N is the number of boxes.
                Each box is represented by a list of coordinates [x1, y1, x2, y2].

        Returns:
        - A NumPy matrix where the element at [i][j] is the IoU between boxes[i] and boxes[j].
        """
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        area = (x2 - x1 + 1) * (y2 - y1 + 1)

        inter_x1 = np.maximum(x1[:, None], x1)
        inter_y1 = np.maximum(y1[:, None], y1)
        inter_x2 = np.minimum(x2[:, None], x2)
        inter_y2 = np.minimum(y2[:, None], y2)

        inter_w = np.maximum(0, inter_x2 - inter_x1 + 1)
        inter_h = np.maximum(0, inter_y2 - inter_y1 + 1)
        inter_area = inter_w * inter_h

        union_area = area[:, None] + area - inter_area

        iou_matrix = inter_area / union_area

        return iou_matrix
    
    def calculate_precision_recall_curve(self, y_true, scores, num_classes):
        """
        Calculate the precision-recall curve for multi-class tasks.

        Parameters:
        - y_true : array-like of shape (n_samples,)
            True labels.
        - scores : array-like of shape (n_samples, n_classes)
            Target scores, can either be probability estimates of each class,
            confidence values, or non-thresholded measure of decisions.
        - num_classes : int
            Number of classes.

        Returns:
        - precision : dict
            Precision values for each class.
        - recall : dict
            Recall values for each class.
        - thresholds : dict
            Decreasing thresholds on the decision function used to compute
            precision and recall for each class.
        """
        y_true_bin = label_binarize(y_true, classes=np.arange(num_classes))
        precision = {}
        recall = {}
        thresholds = {}

        for i in range(num_classes):
            sorted_indices = np.argsort(scores[:, i])[::-1]
            sorted_scores = scores[:, i][sorted_indices]
            sorted_true = y_true_bin[:, i][sorted_indices]
            
            # Append an extra threshold to cover all cases
            thresholds[i] = np.append(sorted_scores, [0], axis=0)
            precision[i] = []
            recall[i] = []
            TP = 0
            FP = 0
            FN = np.sum(y_true_bin[:, i]) # Initial false negatives are all positives

            for j in range(len(sorted_scores)):
                if sorted_true[j] == 1:
                    TP += 1
                    FN -= 1
                else:
                    FP += 1
                
                prec = TP / (TP + FP) if TP + FP > 0 else 0
                rec = TP / (TP + FN) if TP + FN > 0 else 0
                
                precision[i].append(prec)
                recall[i].append(rec)
            
            # Ensure last point is at recall zero
            precision[i].append(1.0)
            recall[i].append(0.0)
        
        return precision, recall, thresholds

    def calculate_map_11_point_interpolated(self, precision_recall_points, num_classes):
        """
        Calculate the mean average precision (mAP) using 11-point interpolation for multi-class tasks.

        Parameters:
        - precision_recall_points: A dictionary where keys are class indices and values are lists of
            tuples representing (recall, precision) points.
        - num_classes: int
            Number of classes.

        Returns:
        - The mAP value as a float.
        """
        mean_average_precisions = []

        for i in range(num_classes):
            points = precision_recall_points[i]
            # Ensure the list is sorted by confidence in descending order
            points = sorted(points, key=lambda x: x[0], reverse=True)
            
            interpolated_precisions = []
            for recall_threshold in [j * 0.1 for j in range(11)]:
                # Find all precisions with recall greater than or equal to the threshold
                possible_precisions = [p for r, p in points if r >= recall_threshold]
                
                # Interpolate precision: take the maximum precision to the right of the current recall level
                if possible_precisions:
                    interpolated_precisions.append(max(possible_precisions))
                else:
                    interpolated_precisions.append(0)
            
            # Calculate the mean of the interpolated precisions
            mean_average_precision = sum(interpolated_precisions) / len(interpolated_precisions)
            mean_average_precisions.append(mean_average_precision)
        
        # Calculate the overall mean average precision
        overall_map = sum(mean_average_precisions) / num_classes
        
        return overall_map

    
if __name__ == "__main__":

    metrics = Metrics()
    # Example usage of calculate_map_11_point_interpolated():
    y_true = np.random.randint(0, 3, 30) # Actual labels for 30 samples with 3 classes (0, 1, 2)
    scores = np.random.rand(30, 3) # Random predicted scores/probabilities for each class
    num_classes = 3
    
    precision, recall, thresholds = metrics.calculate_precision_recall_curve(y_true, scores, num_classes)
    print("Precision:", precision)
    print("Recall:", recall)
    print("Thresholds:", thresholds)

    precision_recall_points = {i: list(zip(recall[i], precision[i])) for i in range(num_classes)}

    map_value = metrics.calculate_map_11_point_interpolated(precision_recall_points, num_classes)
    print(f"Mean Average Precision: {map_value:.4f}")

