## Usage

The system provides a Flask API for interaction. Below are the available endpoints and their usage.

## Installation
1. **Build the docker image**:
`docker build -t object_detection_system .`

2. **Run the Docker container**:
`docker run -p 5000:5000 object_detection_system`

## API Endpoints

### Process Video

-   **Endpoint**: `/process_video`
-   **Method**: `POST`
-   **Description**: Processes a video file for object detection.
- **Sample Curl Command**:
`curl -X POST -F 'file=@/path/to/video.mp4' http://localhost:5000/process_video`

### List Detections

-   **Endpoint**: `/list_detections`
-   **Method**: `GET`
-   **Description**: Lists detections given a specific frame range.
- **Sample Curl Command**:
`curl -X GET "http://localhost:5000/list_detections?start_frame=1&end_frame=100"`

### Get Top Hard Negatives

-   **Endpoint**: `/top_hard_negatives`
-   **Method**: `GET`
-   **Description**: Retrieves the top-N hard negatives from the dataset.
-   **Sample Curl Command**:
`curl -X GET "http://localhost:5000/top_hard_negatives?num=5&criteria=f1_score"`

### Get Images with Detections

-   **Endpoint**: `/images_with_detections`
-   **Method**: `GET`
-   **Description**: Retrieves images with detections and associated predictions given a specific frame range.
- **Sample Curl Command**:
`curl -X GET "http://localhost:5000/images_with_detections?start_frame=1&end_frame=100"`

### Get Image

-   **Endpoint**: `/static/predictions/<filename>`
-   **Method**: `GET`
-   **Description**: Serves images from the predictions directory.
- **Sample Curl Command**:
`curl -X GET "http://localhost:5000/static/predictions/frame_20220101-123456.jpg" --output frame.jpg`