import os
import cv2
import time
from src.inference.video_processing import VideoProcessing
from src.inference.object_detection import YOLOObjectDetector
from src.inference.non_maximal_suppression import NMS

class InferenceService:
    def __init__(self, stream, detector, nms):
        """
        Initialize the InferenceService class.

        Parameters:
        - stream: An instance of the video stream class (VideoProcessing).
        - detector: An instance of the object detection class (YOLOObjectDetector).
        - nms: An instance of the Non-Maximal Suppression (NMS) class.
        """
        self.stream = stream
        self.detector = detector
        self.nms = nms
        self.save_dir = 'object_detection_system/storages/prediction'
        os.makedirs(self.save_dir, exist_ok=True)  # Create the directory if it doesn't exist

    def detect(self):
        """
        Perform object detection on frames from the UDP stream and save frames with detections.
        - Requirement: The system should be able to process and detect objects within videos.
        - Requirement: The system should be able to store images (as a .jpg image) with detections and their associated predictions (in either a PASCAL, YOLO, or COCO format).

        Yields:
        - filtered_detections: The list of filtered detections after applying NMS.
        """
        for frame in self.stream.capture_udp_stream():
            # Predict objects in the frame using the detector
            output = self.detector.predict(frame)
            # Process the output to get class IDs, confidence scores, and bounding boxes
            detections = self.detector.process_output(output)
            # Apply Non-Maximal Suppression to filter detections
            filtered_detections = self.nms.filter(detections)
            if filtered_detections[0]:  # Check if there are any detections
                self._save(frame, filtered_detections)
            # Draw labels on the frame
            self.detector.draw_labels(frame, filtered_detections)
            yield filtered_detections

    def _save(self, frame, detections):
        """
        Save the frame to the specified directory with a unique filename.

        Parameters:
        - frame: The image frame to be saved.
        - detections: The list of detections to be annotated in the image.
        """
        # Generate a timestamp for the filename
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        frame_path = os.path.join(self.save_dir, f"frame_{timestamp}.jpg")
        # Save the frame as a JPEG file
        cv2.imwrite(frame_path, frame)
        print(f"Frame saved at {frame_path}")
        # Save the annotations
        annotation_path = os.path.join(self.save_dir, f"frame_{timestamp}.txt")
        with open(annotation_path, 'w') as f:
            for class_id, confidence, box in zip(*detections):
                x, y, w, h = box
                f.write(f"{class_id} {confidence} {x} {y} {w} {h}\n")

    def process_video(self, video_file):
        """
        Process a video file for object detection and save frames with detections.
        - Requirement: The system must be able to process and detect objects within videos.
        - Requirement: The system should be able to store images (as a .jpg image) with detections and their associated predictions (in either a PASCAL, YOLO, or COCO format).

        Parameters:
        - video_file: The video file to process.

        Returns:
        - detections: The list of detections for each frame in the video.
        """
        video_path = os.path.join(self.save_dir, video_file.filename)
        video_file.save(video_path)
        cap = cv2.VideoCapture(video_path)
        frame_count = 0
        detections = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            # Predict objects in the frame using the detector
            output = self.detector.predict(frame)
            # Process the output to get class IDs, confidence scores, and bounding boxes
            detection = self.detector.process_output(output)
            # Apply Non-Maximal Suppression to filter detections
            filtered_detections = self.nms.filter(detection)
            if filtered_detections[0]:
                self._save(frame, filtered_detections)
            detections.append({'frame': frame_count, 'detections': filtered_detections})

        cap.release()
        return detections

    def list_detections(self, start_frame, end_frame):
        """
        List detections given a specific frame range.
        - Requirement: The system should have an endpoint that lists all the detections given a specific frame range.

        Parameters:
        - start_frame: The starting frame number.
        - end_frame: The ending frame number.

        Returns:
        - detections: A list of detections for each frame in the specified range.
        """
        detections = []
        for frame_num in range(start_frame, end_frame + 1):
            annotation_path = os.path.join(self.save_dir, f"frame_{frame_num}.txt")
            if os.path.exists(annotation_path):
                with open(annotation_path, 'r') as f:
                    frame_detections = [line.strip() for line in f.readlines()]
                    detections.append({f'frame_{frame_num}': frame_detections})
        return detections

    def get_images_with_detections(self, start_frame, end_frame):
        """
        Retrieve images with detections and their associated predictions given a specific frame range.
        - Requirement: The system should have an endpoint to return images with detections and associated predictions given a specific frame range.

        Parameters:
        - start_frame: The starting frame number.
        - end_frame: The ending frame number.

        Returns:
        - images_info: A list of dictionaries containing paths to images and their annotation files.
        """
        images_info = []
        for frame_num in range(start_frame, end_frame + 1):
            image_path = os.path.join(self.save_dir, f"frame_{frame_num}.jpg")
            annotation_path = os.path.join(self.save_dir, f"frame_{frame_num}.txt")
            if os.path.exists(image_path) and os.path.exists(annotation_path):
                images_info.append({
                    'image_path': f'/static/predictions/frame_{frame_num}.jpg',
                    'annotation_path': f'/static/predictions/frame_{frame_num}.txt'
                })
        return images_info

if __name__ == "__main__":
    # Configuration for YOLO model
    cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
    weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
    names_path = "yolo_resources/logistics.names"
    
    # Initialize the YOLO object detector
    detector = YOLOObjectDetector(cfg_path, weights_path, names_path)

    # Parameters for Non-Maximal Suppression
    score_threshold = .5
    iou_threshold = .4
    nms = NMS(score_threshold, iou_threshold)

    # Source for video stream
    video_source = 'udp://127.0.0.1:23000'
    stream = VideoProcessing(video_source)

    # Initialize the inference service
    pipeline = InferenceService(stream, detector, nms)

    # Perform detection on the video stream
    for detections in pipeline.detect():
        print(detections)
