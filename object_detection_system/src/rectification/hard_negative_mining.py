import cv2
import numpy as np
import os
import random
from glob import glob
import pandas as pd


class HardNegativeMiner:
    """
    A class to mine hard negative examples for a given model.

    Attributes:
        model: The model used for prediction.
        nms: Non-maximum suppression object.
        measure: Measure to evaluate predictions.
        dataset_dir: Directory containing the dataset.
        table: DataFrame to store the results.
    """

    def __init__(self, model, nms, measure, dataset_dir):
        """
        Initialize the HardNegativeMiner with model, nms, measure, and dataset directory.

        Args:
            model: The model used for prediction.
            nms: Non-maximum suppression object.
            measure: Measure to evaluate predictions.
            dataset_dir: Directory containing the dataset.
        """
        self.model = model
        self.nms = nms
        self.measure = measure
        self.dataset_dir = dataset_dir
        self.table = pd.DataFrame(columns=['annotation_file', 'image_file'] + self.measure.columns)

    def __read_annotations(self, file_path):
        """
        Read annotations from a text file.

        Args:
            file_path (str): Path to the annotation file.

        Returns:
            list: List of annotations in the format (class_label, x_center, y_center, width, height).
        """
        annotations = []
        with open(file_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                class_label = int(parts[0])
                bbox = list(map(float, parts[1:]))
                annotations.append((class_label, *bbox))
        return annotations

    def __predict(self, image):
        """
        Make a prediction on the provided image using the model.

        Args:
            image (ndarray): The image to predict on.

        Returns:
            output: The model's prediction output.
        """
        output = self.model.predict(image)
        return output

    def __construct_table(self):
        """
        Construct a table with image files, annotation files, and measures.

        This method reads images and annotations, makes predictions,
        computes measures, and appends the results to the table.
        """
        for image_file, annotation_file in zip(
                sorted(glob(os.path.join(self.dataset_dir, "*.jpg"))),
                sorted(glob(os.path.join(self.dataset_dir, "*.txt")))):

            image = cv2.imread(image_file)
            annotation = self.__read_annotations(annotation_file)
            prediction = self.__predict(image)

            measures = self.measure.compute(prediction, annotation)

            self.table = self.table.append(
                {'annotation_file': annotation_file, 'image_file': image_file, **measures},
                ignore_index=True
            )

    def sample_hard_negatives(self, num_hard_negatives, criteria):
        """
        Sample hard negative examples based on the specified criteria.

        Args:
            num_hard_negatives (int): The number of hard negatives to sample.
            criteria (str): The criteria to sort and sample the hard negatives.

        Returns:
            DataFrame: A DataFrame containing the sampled hard negative examples.
        """
        self.__construct_table()
        self.table.sort_values(by=criteria, inplace=True, ascending=False)
        return self.table.head(num_hard_negatives)


class RectificationService:
    def __init__(self, hard_negative_miner, augmentations=None):
        """
        Initialize the RectificationService.

        Args:
            hard_negative_miner: An instance of HardNegativeMiner.
            augmentations: List of augmentation functions to apply to images.
        """
        self.hard_negative_miner = hard_negative_miner
        self.augmentations = augmentations or [self.flip, self.rotate, self.add_noise]

    def select_hard_negatives(self, num_hard_negatives, criteria):
        """
        Select the top-N hard negatives from the dataset.

        Args:
            num_hard_negatives (int): The number of hard negatives to select.
            criteria (str): The criteria for selecting hard negatives.

        Returns:
            DataFrame: A DataFrame containing the sampled hard negatives.
        """
        hard_negatives = self.hard_negative_miner.sample_hard_negatives(num_hard_negatives, criteria)
        return hard_negatives

    def perform_augmentations(self, image):
        """
        Perform augmentations on an image.

        Args:
            image: The input image to augment.

        Returns:
            List: A list of augmented images.
        """
        augmented_images = [aug(image) for aug in self.augmentations]
        return augmented_images

    def flip(self, image):
        """
        Flip the image horizontally.

        Args:
            image: The input image to flip.

        Returns:
            The flipped image.
        """
        return cv2.flip(image, 1)

    def rotate(self, image):
        """
        Rotate the image by 15 degrees.

        Args:
            image: The input image to rotate.

        Returns:
            The rotated image.
        """
        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, 15, 1.0)
        return cv2.warpAffine(image, M, (w, h))

    def add_noise(self, image):
        """
        Add Gaussian noise to the image.

        Args:
            image: The input image to add noise to.

        Returns:
            The image with added noise.
        """
        noise = np.random.normal(0, 25, image.shape)
        noisy_image = cv2.add(image, noise, dtype=cv2.CV_8UC3)
        return noisy_image


if __name__ == "__main__":
    # Example usage:
    from object_detection import YOLOObjectDetector
    from non_maximal_suppression import NMS
    from measure import Measure  # Placeholder for actual measure implementation

    dataset_path = 'path/to/logistics.zip'
    model = YOLOObjectDetector(cfg_path="path/to/yolo.cfg", weights_path="path/to/yolo.weights", names_path="path/to/names")
    nms = NMS(score_threshold=0.5, nms_iou_threshold=0.4)
    measure = Measure(columns=['precision', 'recall', 'f1_score'])

    hard_negative_miner = HardNegativeMiner(model, nms, measure, dataset_path)
    rect_service = RectificationService(hard_negative_miner)

    num_hard_negatives = 5
    criteria = 'f1_score'  # Example criteria for selecting hard negatives
    hard_negatives = rect_service.select_hard_negatives(num_hard_negatives, criteria)
    print(f"Hard negatives: {hard_negatives}")

    for _, row in hard_negatives.iterrows():
        image_path = row['image_file']
        image = cv2.imread(image_path)
        augmented_images = rect_service.perform_augmentations(image)
        for i, aug_image in enumerate(augmented_images):
            aug_image_path = os.path.splitext(image_path)[0] + f"_aug_{i}.jpg"
            cv2.imwrite(aug_image_path, aug_image)
            print(f"Saved augmented image: {aug_image_path}")
