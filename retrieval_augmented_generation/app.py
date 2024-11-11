import os
import json
import time
from flask import Flask, request, jsonify
from pipeline import Pipeline

app = Flask(__name__)
os.chdir(app.root_path)
pipeline = Pipeline()

def log_request(endpoint, input_data, output_data, start_time):
    """
    Logs the API request details, including input, output, and response time.

    Parameters:
    - endpoint (str): The name of the API endpoint being called.
    - input_data (dict): The input data received by the API.
    - output_data (dict): The output data returned by the API.
    - start_time (float): The time when the request was received (used to calculate response time).
    """
    log_file = 'storage/logs/api_log.json'
    end_time = time.time()
    response_time = end_time - start_time    
    # Create a log entry with all relevant details
    log_entry = {
        "endpoint": endpoint,
        "input": input_data,
        "output": output_data,
        "response_time": response_time,  # Log the response time in seconds
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start_time))
    }

    # Load existing logs if the file exists
    if os.path.exists(log_file):
        with open(log_file, 'r') as file:
            try:
                logs = json.load(file)
            except json.JSONDecodeError:
                logs = []
    else:
        logs = []

    # Append the new log entry to the logs
    logs.append(log_entry)

    # Save the updated logs back to the file
    with open(log_file, 'w') as file:
        json.dump(logs, file, indent=4)

@app.route('/precompute', methods=['POST'])
def precompute():
    """
    Precompute embeddings for all documents in the corpus and construct the KDTree.

    This endpoint triggers the pipeline's precompute method, which processes all
    documents in the corpus, generates embeddings, and stores them in the KDTree.

    Returns:
    - JSON response confirming the completion of the precomputation process.
    """
    start_time = time.time()  # Start timing the request
    pipeline.precompute()  # Trigger the precompute process in the pipeline
    output_data = {"message": "Precomputation of embeddings and KDTree construction complete."}
    log_request('precompute', None, output_data, start_time)  # Log the request
    return jsonify(output_data)

@app.route('/search', methods=['POST'])
def search():
    """
    Search for the most relevant sentence chunks based on the provided question.

    This endpoint retrieves the top_k most relevant sentence chunks from the corpus,
    based on the input question.

    Request Data (JSON):
    - question (str): The question to search for.
    - top_k (int, optional): The number of top chunks to retrieve (default is 5).

    Returns:
    - JSON response containing the list of relevant sentence chunks.
    """
    start_time = time.time()  # Start timing the request
    data = request.get_json()  # Get JSON data from the request
    question = data.get("question")
    top_k = data.get("top_k", 5)  # Default to top_k=5 if not provided
    pipeline.top_k = top_k  # Set the pipeline's top_k value
    results = pipeline.search_context(question)  # Perform the search
    output_data = {
        "results": [{"document": r[1]['document'], "chunk": r[1]['chunk'], "text": r[1]['text']} for r in results]
    }
    log_request('search', data, output_data, start_time)  # Log the request
    return jsonify(output_data)

@app.route('/answer', methods=['POST'])
def answer():
    """
    Generate an answer to the provided question using the retrieved context.

    This endpoint retrieves the most relevant context chunks and uses them to generate
    an answer to the input question using the BERT model.

    Request Data (JSON):
    - question (str): The question to generate an answer for.
    - top_k (int, optional): The number of top chunks to retrieve for context (default is 5).

    Returns:
    - JSON response containing the generated answer.
    """
    start_time = time.time()  # Start timing the request
    data = request.get_json()  # Get JSON data from the request
    question = data.get("question")
    top_k = data.get("top_k", 5)  # Default to top_k=5 if not provided
    pipeline.top_k = top_k  # Set the pipeline's top_k value
    answer = pipeline.answer_question(question)  # Generate the answer
    output_data = {"question": question, "answer": answer}
    log_request('answer', data, output_data, start_time)  # Log the request
    return jsonify(output_data)

@app.route('/evaluate', methods=['POST'])
def evaluate():
    """
    Evaluate the system's performance on a series of questions.

    This endpoint compares the model's answers with the provided ground truths
    and returns the evaluation results, including scores and match results.

    Request Data (JSON):
    - questions (list): A list of dictionaries, each containing:
        - question (str): The question to be evaluated.
        - ground_truth (str): The correct answer to the question.

    Returns:
    - JSON response containing the evaluation results for each question.
    """
    start_time = time.time()  # Start timing the request
    data = request.get_json()  # Get JSON data from the request
    questions = data.get("questions")
    results = []
    for q in questions:
        question = q.get("question")
        ground_truth = q.get("ground_truth")
        ground_truth, model_answer, score, match_result = pipeline.evaluate_question(question, ground_truth)
        results.append({
            "question": question,
            "ground_truth": ground_truth,
            "model_anser": model_answer,
            "score": str(score),
            "match_result": str(match_result)
        })
    output_data = {"results": results}
    log_request('evaluate', data, output_data, start_time)  # Log the request
    return jsonify(output_data)

if __name__ == '__main__':
    # Ensure the logs directory exists
    if not os.path.exists('storage/logs'):
        os.makedirs('storage/logs')
    app.run(debug=True)
