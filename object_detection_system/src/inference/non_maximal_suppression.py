import numpy as np
import cv2

class NMS:
    def __init__(self, score_threshold, nms_iou_threshold):
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold

    def filter(self, outputs):
        """
        Perform Non-Maximal Suppression (NMS) on bounding boxes, with class IDs.

        Parameters:
        boxes (list of tuples): List of bounding boxes, each represented as (x1, y1, x2, y2).
        scores (list of floats): List of confidence scores for each bounding box.
        class_ids (list of ints): List of class IDs for each bounding box.
        iou_threshold (float): IoU threshold for NMS.

        Returns:
        list of tuples: The filtered list of bounding boxes, confidence scores, and class IDs after applying NMS.
        """
        class_ids, scores, bboxes = outputs
        indices = cv2.dnn.NMSBoxes(bboxes, scores, self.score_threshold, self.nms_iou_threshold)
        class_ids = [class_ids[i] for i in indices]
        bboxes = [bboxes[i] for i in indices]
        scores = [scores[i] for i in indices]

        return class_ids, scores, bboxes


if __name__ == "__main__":
    from object_detection import YOLOObjectDetector
    import cv2
    # Model arguments
    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"
    score_threshold=.5

    yolo_detector = YOLOObjectDetector(cfg_path, weights_path, names_path)

    iou_threshold = .4
    
    nms = NMS(score_threshold,iou_threshold)
    frame = cv2.imread("yolo_resources/test_images/test_images.jpg")

    output = yolo_detector.predict(frame)
    output = yolo_detector.process_output(output)
    yolo_detector.draw_labels(frame, output)

    output = nms.filter(output)

    yolo_detector.draw_labels(frame, output)
