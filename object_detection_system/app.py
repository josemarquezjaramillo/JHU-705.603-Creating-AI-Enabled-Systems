import os
from flask import Flask, request, jsonify, send_from_directory
from src.inference.video_processing import VideoProcessing
from src.inference.object_detection import YOLOObjectDetector
from src.inference.non_maximal_suppression import NMS
from src.rectification.hard_negative_mining import HardNegativeMiner
from pipeline import InferenceService


app = Flask(__name__)
os.chdir(app.root_path)

# Initialize inference service
cfg_path = "yolo_resources/yolov4-tiny-logistics_size_416_1.cfg"
weights_path = "yolo_resources/models/yolov4-tiny-logistics_size_416_1.weights"
names_path = "yolo_resources/logistics.names"
detector = YOLOObjectDetector(cfg_path, weights_path, names_path)
score_threshold = 0.5
iou_threshold = 0.4
nms = NMS(score_threshold, iou_threshold)
video_source = 'udp://127.0.0.1:23000'
stream = VideoProcessing(video_source)
inference_service = InferenceService(stream, detector, nms)

@app.route('/process_video', methods=['POST'])
def process_video():
    """
    Endpoint to process a video file for object detection.
    - Requirement: The system must be able to process and detect objects within videos.
    - Requirement: The system should be able to store images (as a .jpg image) with 
    detections and their associated predictions (in either a PASCAL, YOLO, or COCO format).

    Request:
    - file: The video file to process.

    Response:
    - JSON containing detections for each frame in the video.
    """
    video_file = request.files['file']
    response = inference_service.process_video(video_file)
    return jsonify(response)

@app.route('/list_detections', methods=['GET'])
def list_detections():
    """
    Endpoint to list detections given a specific frame range.
    - Requirement: The system should have an endpoint that lists all the detections given a 
    specific frame range. Only return the list of detections.

    Query Parameters:
    - start_frame: The starting frame number.
    - end_frame: The ending frame number.

    Response:
    - JSON containing the list of detections for each frame in the specified range.
    """
    start_frame = int(request.args.get('start_frame'))
    end_frame = int(request.args.get('end_frame'))
    detections = inference_service.list_detections(start_frame, end_frame)
    return jsonify(detections)

@app.route('/top_hard_negatives', methods=['GET'])
def top_hard_negatives():
    """
    Endpoint to get the top-N hard negatives from the dataset.
    - Requirement: The system should select the top-N hard negatives within a particular dataset.
    - Requirement: The system should return the top-N hard negatives of a specific dataset. 
    (Return only the base name of the image/annotation file).

    Query Parameters:
    - num: The number of hard negatives to return.
    - criteria: The criteria to sort and sample the hard negatives.

    Response:
    - JSON containing the top-N hard negatives.
    """
    num_hard_negatives = int(request.args.get('num'))
    criteria = request.args.get('criteria')
    dataset_dir = 'data/logistics.zip'
    hard_negative_miner = HardNegativeMiner(detector, nms, criteria, dataset_dir)
    hard_negatives = hard_negative_miner.sample_hard_negatives(num_hard_negatives, criteria)
    return jsonify(hard_negatives.to_dict('records'))

@app.route('/images_with_detections', methods=['GET'])
def images_with_detections():
    """
    Endpoint to get images with detections and associated predictions given a specific frame range.
    - Requirement: The system should have an endpoint to return images with detections and associated 
    predictions given a specific frame range. Return the images with bounding boxes and the annotated text files.

    Query Parameters:
    - start_frame: The starting frame number.
    - end_frame: The ending frame number.

    Response:
    - JSON containing paths to images with detections and their annotation files.
    """
    start_frame = int(request.args.get('start_frame'))
    end_frame = int(request.args.get('end_frame'))
    images_info = inference_service.get_images_with_detections(start_frame, end_frame)
    return jsonify(images_info)

@app.route('/static/predictions/<filename>', methods=['GET'])
def get_image(filename):
    """
    Endpoint to serve images from the predictions directory.

    Path Parameters:
    - filename: The name of the image file to retrieve.

    Response:
    - The requested image file.
    """
    return send_from_directory('static/predictions', filename)

if __name__ == "__main__":
    # Run the Flask application
    app.run(host='0.0.0.0', port=5000)
