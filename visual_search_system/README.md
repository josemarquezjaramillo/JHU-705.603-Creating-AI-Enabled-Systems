# Visual Identification System

## Overview

The Visual Identification System is designed for IronClad Technology, a defense company requiring robust access control. Traditional ID cards and manual security checks are becoming untenable due to growth, putting stress on security personnel to maintain access. This system leverages visual identification to verify the identity of personnel, ensuring a high identification rate of employees and a low false positive rate of non-employees (intruders). It adapts to different brightness conditions and allows for dynamic adjustment of personnel access.

## Features

- **High Identification Rate**: Achieves a high identification rate of employees.
- **Low False Positive Rate**: Minimizes the false positive rate of non-employees.
- **Adaptability**: Adapts to different brightness conditions.
- **Dynamic Access Control**: Allows for dynamic adjustment of personnel access, including adding and removing personnel.
- **Scalability**: Capable of identifying millions of known personnel.

## System Architecture

### Extraction

1. **Pre-Processing Module**: Formats images for ingestion by the model.
2. **Embedding Module**: Extracts embeddings from images to compute similarities.
3. **Indexing Module**: Indexes each image in the catalog storage and organizes the embedding, allowing the system to scale to millions of employees in the catalog.

### Retrieval

- **Search Module**: Matches probe images with identities in the catalog.

### Interface

- **Interface Module**: 
  - Triggers the authentication process of the probe.
  - Adds or removes identities from the gallery.
  - Audits access history.
  - Reviews access logs to evaluate system performance.

## Directory and File Architecture

```
visual_search_system/
│
├── simclr_resources/
│ ├── model_size_064_resnet_018.pth
│ ├── model_size_064_resnet_034.pth
│ ├── model_size_224_resnet_018.pth
│ └── model_size_224_resnet_034.pth
│ └── probes/
│
├── src/
│ ├── extraction/
│ │ ├── model.py
│ │ └── preprocess.py
│ ├── search/
│ │ ├── indexing.py
│ │ └── search.py
│ └── metrics.py
│
├── storage/
│ ├── embedding/
│ └── gallery/
│
├── pipeline.py
├── main.py
└── Dockerfile
```

## Usage

### Precompute Embeddings

To precompute the embeddings for all images in the gallery and build the KD-Tree:

`python pipeline.py`

### Search Gallery

To search the gallery with a probe image:

```
# Example of searching the gallery
probe_image_path = "simclr_resources/probes/Aaron_Sorkin/Aaron_Sorkin_0002.jpg"
results = pipeline.search_gallery(probe_image_path)
print("Nearest neighbors:")
if results:
    for result in results:
        print(result)
else:
    print("No neighbors found.")
```

### Add a new image

To add a new image to the KD-Tree and update the index:

```
new_image_path = "storage/gallery/Mark_Warner/Mark_Warner_0001.jpg"
pipeline.add_image(new_image_path)
```

## Interface Service (Flask Application)

### Running the Flask Application

To run the flask application:
` python main.py `

### API Endpoints

-   **Authenticate**: `POST /authenticate`
    
    -   Trigger the authentication process with a probe image.
    -   Request: `multipart/form-data` with an image file.
    -   Response: JSON with the predicted identities.
    -   Sample `curl` command:        
        `curl -X POST http://localhost:5000/authenticate \
          -F "image=@/path/to/your/probe_image.jpg"` 
        
-   **Add Identity**: `POST /add_identity`
    
    -   Add a new identity to the gallery.
    -   Request: `multipart/form-data` with an image file.
    -   Response: JSON with the status and image name.
    -   Sample `curl` command:        
        `curl -X POST http://localhost:5000/add_identity \
          -F "image=@/path/to/your/new_identity_image.jpg"` 
        
-   **Remove Identity**: `POST /remove_identity`
    
    -   Remove an identity from the gallery.
    -   Request: JSON with the first name, last name, and image filename.
    -   Response: JSON with the status and image name.
    -   Sample `curl` command:
        `curl -X POST http://localhost:5000/remove_identity \
          -H "Content-Type: application/json" \
          -d '{
                "first_name": "Mark",
                "last_name": "Warner",
                "image_filename": "Mark_Warner_0001"
              }'` 
        
-   **Access Logs**: `GET /access_logs`
    
    -   Retrieve the access log history.
    -   Request: Query parameters for the specific time period.
    -   Response: JSON with the access logs.
    -   Sample `curl` command:
        `curl -X GET "http://localhost:5000/access_logs?start_date=2024-01-01&end_date=2024-12-31"`
        
-   **Change Model**: `POST /change_model`
    
    -   Change the model architecture and image size.
    -   Request: JSON with the new image size and architecture.
    -   Response: JSON with the status of the model change.
    -   Sample `curl` command:        
        `curl -X POST http://localhost:5000/change_model \
          -H "Content-Type: application/json" \
          -d '{
                "image_size": 64,
                "architecture": "resnet_018"
              }'`

## Containerization

### Docker

To build and run the Docker container:

1.  Build the Docker image:
    
    `docker build -t visual-identification-system .` 
    
2.  Run the Docker container:
    
    `docker run -p 5000:5000 visual-identification-system`

