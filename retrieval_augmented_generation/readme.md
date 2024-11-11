
# Retrieval-Augmented Generation (RAG) Pipeline  

This project implements a Retrieval-Augmented Generation (RAG) system for question answering. The system combines document retrieval with a pre-trained BERT model to generate accurate answers based on retrieved context from a large corpus.  

## Project Overview  

The pipeline consists of three main services:

1.  **Extraction Service**: Handles document preprocessing, including chunking and embedding generation.

2.  **Retrieval Service**: Uses KDTree for efficient retrieval of relevant context chunks based on the query.

3.  **Generator Service**: Generates answers using a pre-trained BERT model, leveraging the retrieved context.

  

## Project Structure

  

```

retrieval_augmented_generation/

│

├── app.py # Flask application serving the API

├── pipeline.py # Main pipeline implementation

├── metrics.py # Metrics for evaluating model performance

├── src/

│ ├── extraction/

│ │ ├── embedding.py # Embedding generation using Sentence Transformers

│ │ ├── preprocessing.py # Document processing (chunking, trimming)

│ ├── retrieval/

│ │ ├── index.py # KDTree implementation

│ │ ├── search.py # KDTree search for nearest neighbors

│ ├── generator/

│ ├── question_answering.py # BERT model for question answering

│ └── storage/

├── corpus/ # Storage for document corpus

├── embeddings/ # Storage for precomputed embeddings

└── logs/ # Storage for request logs

```

  

## Setup Instructions

  

1.  **Clone the Repository**

  

```bash

git clone https://github.com/creating-ai-enabled-systems-summer-2024/marquezjaramillo-jose.git

cd retrieval-augmented-generation

```

  

-  **Create and Activate a Virtual Environment**

```python3 -m venv venv```

```source venv/bin/activate ``

-  **Install Dependencies**

`pip install -r requirements.txt`

-  **Run the Flask API**

`python app.py`

The Flask app will start running on `http://127.0.0.1:5000`.

## API Endpoints
  

### 1. **Precompute Embeddings and KDTree Construction**

  

-  **Endpoint**: `/precompute`

-  **Method**: `POST`

-  **Description**: This endpoint precomputes embeddings for all documents in the corpus and constructs a KDTree for efficient retrieval.

-  **Sample Request**:

`curl -X POST http://127.0.0.1:5000/precompute`

-  **Sample Response**:

```
{"message": "Precomputation of embeddings and KDTree construction complete."}
```

  

### 2. **Search Context for a Question**

  

-  **Endpoint**: `/search`

-  **Method**: `POST`

-  **Description**: This endpoint retrieves the most relevant sentence chunks for a given question using the KDTree.

-  **Sample Request**:

```bash
curl -X POST http://127.0.0.1:5000/search -H "Content-Type: application/json" -d '{

"question": "What is the capital of France?", top_k": 5}'
```

-  **Sample Response**:

```bash
{

"results": [

{

"document": "document_1.txt",

"chunk": 0,

"text": "The capital of France is Paris."

},

{

"document": "document_1.txt",

"chunk": 1,

"text": "Paris is a major European city and a global center for art, fashion, and culture."

}

// Additional chunks...

]

}
```

  

### 3. **Generate Answer for a Question**

  

-  **Endpoint**: `/answer`

-  **Method**: `POST`

-  **Description**: This endpoint generates an answer to a given question by retrieving the most relevant context and using the BERT model to derive the answer.

-  **Sample Request**:

```bash
curl -X POST http://127.0.0.1:5000/answer -H "Content-Type: application/json" -d '{

"question": "What is the capital of France?", "top_k": 5}'
```

-  **Sample Response**:

```bash
{
"question": "What is the capital of France?",
"answer": "The capital of France is Paris."
}
```

  

### 4. **Evaluate the Model on a Series of Questions**

  

-  **Endpoint**: `/evaluate`

-  **Method**: `POST`

-  **Description**: This endpoint evaluates the model's performance on a series of questions by comparing the model's answers with the ground truths.

-  **Sample Request**:

```bash
curl -X POST http://127.0.0.1:5000/evaluate -H "Content-Type: application/json" -d '{
"questions": [
{"question": "What is the capital of France?", "ground_truth": "Paris"},
{"question": "Who was the first president of the United States?",
"ground_truth": "George Washington"}
]}
```

-  **Sample Response**:

```bash
{"results": [
{
"question": "What is the capital of France?",
"ground_truth": "Paris",
"score": 0.95,
"match_result": true
},
{
"question": "Who was the first president of the United States?",
"ground_truth": "George Washington",
"score": 0.90,
"match_result": true
}
]
}
```

  

## Design Considerations

  

### 1. **Sentence Chunking (`sentences_per_chunk`)**

  

This parameter determines the number of sentences per chunk when splitting documents. It directly impacts the granularity of the context retrieved. Smaller chunks might yield higher precision but could lose broader context, while larger chunks provide more context but might dilute relevance.

  

### 2. **Top-k Retrieval (`top_k`)**

  

The `top_k` parameter defines the number of the most relevant chunks retrieved for a given question. Increasing `top_k` provides more context but might introduce irrelevant information, increasing computational overhead.

  

### 3. **Embedding Model Choice**

  

The choice of embedding model affects how well the text is represented in the embedding space, impacting retrieval accuracy and generation quality.

  

### 4. **KDTree for Efficient Retrieval**

  

The KDTree structure is critical for scaling retrieval operations as the corpus size grows. It ensures efficient and quick access to relevant context.

  

### 5. **Context and Answer Generation Quality**

  

The BERT model's ability to generate accurate answers depends on how well it utilizes the retrieved context. The system's overall success hinges on producing contextually rich and accurate responses.

  

## Results and Analysis

  

The results of various design considerations (such as `sentences_per_chunk` and `top_k`) indicate that the system's performance is sensitive to these parameters. Optimal values need to balance precision, context richness, and computational efficiency.

  

## Conclusion

  

This Retrieval-Augmented Generation pipeline provides a scalable and efficient solution for question answering, combining document retrieval with a state-of-the-art BERT model. The system is highly configurable, allowing for fine-tuning based on the specific requirements of the use case.

  

### Example Workflow

  

Here's a step-by-step example workflow using the API endpoints:

  

1.  **Precompute Embeddings:**

```bash

curl -X POST http://localhost:5000/precompute

```

- Prepares the system by processing all documents in the corpus, creating embeddings, and constructing the KDTree.

  

2.  **Add a New Document:**

```bash
curl -F "document_path=/path/to/new_document.txt" http://localhost:5000/add_document
```

- Adds a new document to the corpus, processes it, and updates the KDTree.

  

3.  **Search for Relevant Information:**

```bash
curl -X POST http://localhost:5000/search -H "Content-Type: application/json" -d '{"question": "What is the capital of France?", "top_k": 3}'
```

- Searches the corpus for the top 3 most relevant chunks of text related to the question.

  

4.  **Remove an Outdated Document:**

```bash
curl -F "document_path=/path/to/old_document.txt" http://localhost:5000/remove_document
```

- Removes an outdated or irrelevant document from the corpus, updating the KDTree accordingly.