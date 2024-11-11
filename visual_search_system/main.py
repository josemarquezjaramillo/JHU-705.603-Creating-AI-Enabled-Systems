from flask import Flask, request, jsonify
from pipeline import Pipeline
from src.extraction.model import Model
from src.extraction.preprocess import Preprocessing
from src.search.indexing import KDTree
from src.search.search import KDTreeSearch, Measure
import os
import json
from datetime import datetime

app = Flask(__name__)
os.chdir(app.root_path)
# Location of storage
GALLERY_STORAGE = os.path.join('storage', 'gallery')
EMBEDDINGS_STORAGE = os.path.join('storage', 'embedding')
ACCESS_LOGS_STORAGE = os.path.join('storage', 'access_logs')

# Directory for storing logs
if not os.path.exists(ACCESS_LOGS_STORAGE):
    os.makedirs(ACCESS_LOGS_STORAGE)

def log_access(action, details):
    """
    Log access actions to a JSON file.

    Args:
        action (str): The action being logged (e.g., "authenticate", "add_identity").
        details (dict): Additional details to log.
    """
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "details": details
    }
    log_filename = os.path.join(ACCESS_LOGS_STORAGE, f"{datetime.now().strftime('%Y-%m-%d')}.json")
    with open(log_filename, 'a') as log_file:
        log_file.write(json.dumps(log_entry) + '\n')

def initialize_pipeline(image_size, architecture):
    """
    Initialize the pipeline with the given image size and architecture.

    Args:
        image_size (int): The image size for preprocessing.
        architecture (str): The model architecture.

    Returns:
        Pipeline: The initialized pipeline instance.
    """
    os.chdir('visual_search_system/')
    model_name = f"model_size_{str(image_size).zfill(3)}_{architecture}"
    preprocessing = Preprocessing(image_size=image_size)
    model = Model(f"simclr_resources/{model_name}.pth")
    index = KDTree(k=256, points=[])
    search_euclidean = KDTreeSearch(index, Measure.euclidean)
    pipeline = Pipeline(preprocessing=preprocessing,
                        model=model,
                        index=index,
                        search=search_euclidean)

    # Precompute the embeddings and build the KD-Tree
    pipeline.precompute()
    return pipeline

# Initialize the pipeline with default values
image_size = 224
architecture = 'resnet_034'
pipeline = initialize_pipeline(image_size, architecture)

@app.route('/authenticate', methods=['POST'])
def authenticate():
    """
    Authenticate endpoint to trigger the authentication process.
    
    This endpoint takes an image file as input and returns the k-nearest neighbors 
    (identities) from the gallery.
    
    Request:
        multipart/form-data with an image file.
        
    Response:
        JSON with the predicted identities.
    """
    data = request.files['image']
    image_path = os.path.join("uploads", data.filename)
    data.save(image_path)
    results = pipeline.search_gallery(image_path)
    os.remove(image_path)
    
    log_access("authenticate", {"image": data.filename, "results": results})
    
    return jsonify(results)

@app.route('/add_identity', methods=['POST'])
def add_identity():
    """
    Add Identity endpoint to add a new identity to the gallery.
    
    This endpoint takes an image file, saves it to the gallery, 
    and updates the KD-Tree with the new identity.
    
    Request:
        multipart/form-data with an image file.
        
    Response:
        JSON with the status and image name.
    """
    data = request.files['image']
    image_path = os.path.join("storage", "gallery", data.filename)
    data.save(image_path)
    pipeline.add_image(image_path)
    
    log_access("add_identity", {"image": data.filename})
    
    return jsonify({"status": "added", "image": data.filename})

@app.route('/remove_identity', methods=['POST'])
def remove_identity():
    """
    Remove Identity endpoint to remove an identity from the gallery.
    
    This endpoint takes JSON data with the first name, last name, and image filename, 
    removes the corresponding embedding, and updates the KD-Tree.
    
    Request:
        JSON with the first name, last name, and image filename.
        
    Response:
        JSON with the status and image name.
    """
    data = request.json
    first_name = data['first_name']
    last_name = data['last_name']
    image_filename = data['image_filename']
    embedding_filename = os.path.join('visual_search_system/storage/embedding', pipeline.model_name, f"{first_name}_{last_name}", f"{image_filename}.npy")
    if os.path.exists(embedding_filename):
        os.remove(embedding_filename)
        pipeline.precompute()  # Rebuild the KD-Tree
        
        log_access("remove_identity", {"first_name": first_name, "last_name": last_name, "image_filename": image_filename})
        
        return jsonify({"status": "removed", "image": image_filename})
    else:
        return jsonify({"status": "error", "message": "Image not found"}), 404

@app.route('/access_logs', methods=['GET'])
def access_logs():
    """
    Access Logs endpoint to retrieve the access log history.
    
    This endpoint returns access logs for a specific time period.
    
    Request:
        Query parameters for the specific time period: start_date, end_date.
        
    Response:
        JSON with the access logs.
    """
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    logs = []

    for log_file in os.listdir('visual_search_system/storage/access_logs'):
        log_date_str = log_file.replace('.json', '')
        if start_date <= log_date_str <= end_date:
            with open(os.path.join('visual_search_system/storage/access_logs', log_file), 'r') as lf:
                logs.extend([json.loads(line) for line in lf])

    return jsonify(logs)

@app.route('/change_model', methods=['POST'])
def change_model():
    """
    Change Model endpoint to change the model architecture and image size.
    
    This endpoint takes JSON data with the new image size and architecture, 
    updates the model and preprocessing pipeline, and recomputes the embeddings.
    
    Request:
        JSON with the new image size and architecture.
        
    Response:
        JSON with the status of the model change.
    """
    data = request.json
    image_size = data['image_size']
    architecture = data['architecture']
    global pipeline
    pipeline = initialize_pipeline(image_size, architecture)
    
    log_access("change_model", {"image_size": image_size, "architecture": architecture})
    
    return jsonify({"status": "changed", "model": f"model_size_{str(image_size).zfill(3)}_{architecture}"})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
