This directory contains all of the files required for the Fraud Detection System case study.  

This package has been created so that it can be run an executed using flask or in a docker container.
  

# Set Up

## Setting up the data
Before running the system, you need to place some datafiles in the `data/` folder, otherwise the system will not have any data to source when creating a `DataEngineering()` instance. The data formats supported are `.json, .csv, .parquet, .xlsx`. The structure of the files must follow the specifications provided in the case study definition. 

## Requirements

The requirements.txt file will indicate all of the packages required to succesfully execute the code.  

## Dockerfile

The file Dockerfile will provide the necessarily files to initiate build a docker image and run docker containers  

# Sample curl endpoint calls

 1. Generating a new dataset: `curl -X POST "http://localhost:5000/generate_new_dataset?version=sampled_dataset_abc&size=10000"`

 2. Getting the description of a dataset: `curl -X GET "http://localhost:5000/dataset_description?version=1.0"`

 3. Ingesting external data into the data folder: `curl -X PUT http://localhost:5000/ingest_dataset -F "file=@C:/trash/sample.csv" -H "Content-Type: multipart/form-data"`

 4. List all available Models: `curl -X GET "http://localhost:5000/list_models"`

 5. List all avialable sampled datasets: `curl -X GET "http://localhost:5000/list_datasets"`
 6. Predicting a label
	 1. Using an external file: `curl -X POST http://localhost:5000/predict -F "file=@C:/trash/sample.json" -F "model_name=random_forest_classifier" -H "Content-Type: multipart/form-data"`
	 2. Using a JSON dictionary `curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d "{"Unnamed: 0":{"304862":304862},"trans_date_trans_time":{"304862":"2019-05-27 21:25:06"},"cc_num":{"304862":30487648872433},"merchant":{"304862":"fraud_Schoen Ltd"},"category":{"304862":"kids_pets"},"amt":{"304862":29.7},"first":{"304862":"Stephanie"},"last":{"304862":"Crane"},"sex":{"304862":null},"street":{"304862":"144 Martinez Curve"},"city":{"304862":"Central"},"state":{"304862":"IN"},"zip":{"304862":47110.0},"lat":{"304862":38.097},"long":{"304862":-86.1723},"city_pop":{"304862":null},"job":{"304862":"COUNSELLOR"},"dob":{"304862":"05\/01\/1955"},"trans_num":{"304862":"f313e27bb5fab3a4b2e5d4d26b8d8c84"},"unix_time":{"304862":1338153906.0},"merch_lat":{"304862":37.434876},"merch_long":{"304862":-86.548949},"is_fraud":{"304862":0.0}}" -G -d "model_name=your_model_name"`
 7. Train a model
	 1. Specify model and sample dataset: `curl -X POST http://localhost:5000/train_model -F "model_name=random_forest_classifier" -F "dataset=sampled_dataset_abc"`
	 2. Specify only a model and sample size - a new sampled dataset will be created: `curl -X POST http://localhost:5000/train_model -F "model_name=random_forest_classifier" -F "sample_size=10000"`