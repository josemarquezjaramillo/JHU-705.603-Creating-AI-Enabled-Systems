import os
import json
import datetime
import pickle
from flask import Flask, request, jsonify
from deployment import DeploymentPipeline
import numpy as np
from data_engineering import DataEngineering

app = Flask(__name__)
os.chdir(app.root_path)
datasets_dir = os.path.join(app.root_path, 'datasets')
data_dir = os.path.join(app.root_path, 'data')
models_dir = os.path.join(app.root_path, 'models')
entries_dir = os.path.join(app.root_path, 'entries')
os.makedirs(entries_dir, exist_ok=True)

def initialize_pipeline():
    """
    Initializes the DeploymentPipeline with all dataset files in the datasets directory.
    """
    data_files = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
    return DeploymentPipeline(data_files)

pipeline = initialize_pipeline()

def list_directory_names_and_paths(directory):
    """
    Process a list of filenames to create a dictionary where keys are the filenames
    without the file extension, and values are the original filenames.

    Args:
        filenames (list): A list of filenames as strings.

    Returns:
        dict: A dictionary with filenames without extensions as keys, and full filenames as values.
    """
    result_dict = {}
    for filename in os.listdir(directory):
        # Extract the base filename without the extension
        base_name = os.path.splitext(filename)[0]
        # Store in the dictionary: key is the base name, value is the original filename
        result_dict[base_name] = os.path.join(directory,filename)

    return result_dict

@app.route('/')
def index():
    """
    Index route that returns a welcome message.

    Returns:
    JSON response with a welcome message.
    """
    return "Hello World!"


@app.route('/generate_new_dataset', methods=['POST'])
def generate_new_dataset():
    """
    Generates a new dataset based on the specified version and optional size.

    Retrieves the version of the dataset from the URL query parameters and, optionally,
    the size of the dataset to generate. It then triggers the dataset generation process
    within a pipeline and logs the creation in a JSON file.

    URL Parameters:
        version (str): The version identifier for the dataset.
        size (int, optional): The number of samples to include in the dataset (default is 1000).

    Returns:
        JSON response: Contains a success message and a brief description of the dataset.
                       Returns an error message if no version is specified.
    """
    # Extract 'version' from the request's query parameters
    dataset_version = request.args.get('version')
    
    # Extract 'size' from the request's query parameters or use default of 1000
    if 'size' in request.args:
        try:
            sample_size = int(request.args.get('size'))
        except ValueError:  # Handle case where 'size' is not an integer
            return jsonify({"error": "Size must be an integer"}), 400
    else:
        sample_size = 1000
    
    # Check if the 'version' parameter was provided
    if not dataset_version:
        return jsonify({"error": "No version specified"}), 400

    # Generate the dataset using the specified version and sample size
    description = pipeline.generate_new_dataset(os.path.join(datasets_dir, dataset_version), n=sample_size)
    
    # Log the dataset generation event
    pipeline.log(
        'datasets',
        log_entry={
            "version": dataset_version,
            "timestamp": datetime.datetime.now().isoformat(),
            "description": description
        },
        log_file=f'{dataset_version}.json'
    )

    # Respond with success message and dataset description
    return jsonify({"message": f"Generated new dataset {dataset_version}", "description": description}), 200


@app.route('/dataset_description', methods=['GET'])
def get_dataset_description():
    """
    Retrieves a description of a dataset based on a specified version.

    This route queries an underlying logging system or database to retrieve metadata and
    statistical information about a dataset version specified in the URL parameters. This
    can include size, creation date, and other relevant metrics.

    URL Parameters:
        version (str): The identifier for the dataset version to retrieve.

    Returns:
        JSON response: If successful, returns a dictionary containing the dataset description.
                       If no version is specified, returns an error with status code 400.
                       If the dataset description cannot be found, returns an error with status code 404.
    """
    # Fetch 'version' from the query parameters
    dataset_version = request.args.get('version')
    
    # Check if the 'version' parameter was provided
    if not dataset_version:
        return jsonify({"error": "No version specified"}), 400  # Return an error if no version is provided
    
    try:
        # Attempt to retrieve the dataset description from the pipeline logs
        description = pipeline.get_log('datasets', dataset_version)
    except FileNotFoundError:
        # Handle the case where the log file does not exist
        return jsonify({"error": "Description not found"}), 404
    
    # Return the description if found
    return jsonify({"description": description}), 200


@app.route('/ingest_dataset', methods=['PUT'])
def ingest_dataset():
    """
    Ingests a dataset file into the system and updates the data processing pipeline.

    This route allows for the uploading and processing of a file in specific formats (CSV, Parquet, XLSX, JSON).
    It saves the file to a designated directory, processes it using the DataEngineering class, and logs the
    operation along with a description of the dataset.

    Returns:
        JSON: Contains a message confirming the ingestion and details of the file processed,
              including a brief data description and a timestamp.
    """
    # Attempt to retrieve the file from the PUT request
    file = request.files.get('file')
    if file is None:
        return jsonify({"error": "No file specified"}), 400  # File not found in request

    filename = file.filename
    file_path = os.path.join(data_dir, filename)  # data_dir should be defined in your application's config or globally

    # Extract the file type from the filename extension
    file_type = filename.split('.')[-1]
    # Validate the file format
    if file_type not in ['csv', 'parquet', 'xlsx', 'json']:
        return jsonify({"error": "The file needs to be a formatted csv, parquet, xlsx or json file"}), 404

    # Save the file to the designated directory
    file.save(file_path)

    # Process the file using the DataEngineering class
    file_dataset = DataEngineering(file_path)

    # Re-initialize the pipeline with the new dataset, if necessary
    global pipeline
    pipeline = initialize_pipeline()

    # Prepare a log entry for the operation
    log_entry = {
        "filename": filename,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # Update the log entry with a description of the dataset
    log_entry.update(file_dataset.describe())

    # Log the data ingestion event
    pipeline.log(
        'data',
        log_entry=log_entry,
        log_file=f'{filename}.json'
    )

    # Build the response message
    message = {'message': f"{filename} has been ingested into the pipeline"}
    message.update(log_entry)

    return jsonify(message), 200

@app.route('/list_models', methods=['GET'])
def list_models():
    """
    List all trained machine learning models stored as .sav files in the models directory.
    Provides comprehensive details about each model, including type, parameters, performance metrics,
    and other relevant attributes.

    Returns:
        JSON: A list of models with detailed attributes such as model type, parameters, performance metrics,
              feature importances, and other relevant information.
    """
    models_dict = list_directory_names_and_paths(models_dir)
    models = {}
    for model_name in models_dict.keys():
        model_info = None
        model_path = models_dict[model_name]
        with open(model_path, 'rb') as model_file:
            model = pickle.load(model_file)
        file_stats = os.stat(model_path)
        last_modified_date = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        model_info = {
            'last_modified_date': str(last_modified_date),
            'type': str(type(model).__name__),
            'parameters': str(model.get_params()),
            'file_size': str(os.path.getsize(model_path))
        }
        models[model_name] = model_info
    return jsonify(models), 200

@app.route('/list_datasets', methods=['GET'])
def list_datasets():
    """
    List all datasets stored in the datasets directory.
    Provides basic details about each dataset, including file size and last modified date.

    Returns:
        JSON: A list of datasets with detailed attributes such as file size and last modified date.
    """
    datasets = {}
    if len(os.listdir(datasets_dir)) == 0:
        return jsonify({'message':'There are no sampled datasets in the datasets directory'}), 200
    # List all files in the datasets directory
    for filename in os.listdir(datasets_dir):
        # Filter for specific file types if necessary
        file_path = os.path.join(datasets_dir, filename)
        file_stats = os.stat(file_path)
        last_modified_date = datetime.datetime.fromtimestamp(file_stats.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
        datasets[filename] = {
            'file_size': str(os.path.getsize(file_path)),
            'last_modified_date': str(last_modified_date)
        }

    return jsonify(datasets), 200

@app.route('/predict', methods=['POST'])
def predict():
    """
    Route to make predictions based on input data and a specified model.

    The function supports input either as a JSON object directly in the request body, as a JSON file uploaded,
    or as part of form data. Additionally, clients can specify which model to use by providing a model_name parameter.

    Request Parameters:
        model_name (str, optional): The name of the model to use for predictions. Defaults to a standard model if not specified.

    Returns:
        JSON response: Contains the prediction result or an error message.
    """
    # Extract model name from form data or query string
    model_name = request.form.get('model_name', default='random_forest_classifier')  # You can set a default model

    models_dict = list_directory_names_and_paths(models_dir)

    if model_name not in models_dict:
        return jsonify({"error": "Invalid Model Name, please refer to the endpoint /list_models"}), 400

    # Check if a file is part of the request or get JSON data from the request body
    if 'file' in request.files:
        file = request.files['file']
        if file:
            data = json.load(file)
    else:
        data = request.get_json()

    # Validate the presence of data
    if not data:
        return jsonify({"error": "No data provided"}), 400

    # Save the data to a JSON file in the 'entries' directory
    timestamp = datetime.datetime.now()
    json_filename = f"input_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    json_file_path = os.path.join(entries_dir, json_filename)
    with open(json_file_path, 'w') as json_file:
        json.dump(data, json_file)

    # Perform the prediction using the specified model
    model_prediction = pipeline.predict(json_file_path, model_name)
    
    # Convert numpy arrays to lists if necessary
    if isinstance(model_prediction, np.ndarray):
        model_prediction = model_prediction.tolist()
    elif isinstance(model_prediction, dict):
        # If the prediction includes numpy arrays inside a dictionary
        for key, value in model_prediction.items():
            if isinstance(value, np.ndarray):
                model_prediction[key] = value.tolist()

    # Logging the prediction
    pipeline.log(
        'predictions', 
        log_entry={
            'input_data': json_file_path,
            'model_used': model_name,
            'time': timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'prediction': model_prediction
        }, 
        log_file=json_filename
    )    

    return jsonify({"prediction": model_prediction}), 200



@app.route('/train_model', methods=['POST'])
def train_model():
    """
    Train a machine learning model with the specified dataset and parameters.

    The function reads model name, dataset name, and sample size from the POST request.
    It then checks for dataset and model validity, potentially generates a new dataset,
    trains the model, computes metrics, and logs the process.

    Returns:
        JSON: A JSON response with training results or error messages.
    """

    # Extract model name from form data or query string
    model_name = request.form.get('model_name', default='random_forest_classifier')
    # Extract dataset name from form data or query string
    dataset = request.form.get('dataset', default=None)
    # Extract sample size from form data or query string
    sample_size = int(request.form.get('sample_size', 10000))

    timestamp = datetime.datetime.now()

    # Attempt to retrieve the list of available datasets
    datasets_dict = list_directory_names_and_paths(datasets_dir)
    models_dict = list_directory_names_and_paths(models_dir)

    # Validate the provided dataset and model names
    if dataset not in datasets_dict and dataset is not None:
        return jsonify({"error": "Invalid Dataset Name, please refer to the endpoint /list_datasets"}), 400
    elif model_name not in models_dict:
        return jsonify({"error": "Invalid Model Name, please refer to the endpoint /list_models"}), 400
    elif dataset is None:
        dataset =  f'automated_{timestamp.strftime("%Y_%m_%d")}'
        dataset_path = os.path.join(datasets_dir, dataset)
        description = pipeline.generate_new_dataset(dataset_path, n=sample_size)
        pipeline.log(
            'datasets',
            log_entry={
                "version": dataset,
                "timestamp": datetime.datetime.now().isoformat(),
                "description": description
            },
            log_file=f'{dataset}.json'
        )
        datasets_dict = list_directory_names_and_paths(datasets_dir)
        dataset_path = datasets_dict[dataset] 
    else:
        dataset_path = datasets_dict[dataset]        

    # Train the model using the dataset
    new_model_name = f'{model_name}_{timestamp.strftime("%Y_%m_%d")}'
    pipeline.train_model(dataset_path, model_name, new_model_name)

    # Run metrics after training
    metrics = pipeline.metrics.run_metrics()

    # Log training and metrics information
    log_entry = {
        "model_name": new_model_name,
        "timestamp": timestamp.isoformat(),
        "training_data_metrics": metrics,
        "message": f"The model {new_model_name} has been trained and evaluated."
    }
    pipeline.log(
        'models',
        log_entry=log_entry,
        log_file=f'{new_model_name}.json'
    )

    # Return the log entry as a JSON response
    return jsonify(log_entry), 200


if __name__ == '__main__':
    folder = 'data/'
    data_files = [folder+file for file in os.listdir(folder)]
    pipeline = DeploymentPipeline(data_files)
    # print(pipeline.dataset.dataset.shape[0])
    app.run(host='0.0.0.0', port=5000, debug=True)