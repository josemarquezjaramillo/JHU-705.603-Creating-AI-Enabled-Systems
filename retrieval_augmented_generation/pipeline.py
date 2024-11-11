import os
import numpy as np
from glob import glob
from src.extraction.embedding import Embedding
from src.extraction.preprocessing import DocumentProcessing
from src.retrieval.index import KDTree
from src.retrieval.search import KDTreeSearch
from src.generator.question_answering import BERTQuestionAnswer
from metrics import Metrics_Automated

class Pipeline:
    """
    A class used to represent the retrieval-augmented generation pipeline for 
    question answering. This pipeline handles the preprocessing of documents, 
    encoding of text into embeddings, retrieval of relevant context using a KDTree, 
    and generation of answers using a pre-trained BERT model.

    Attributes
    ----------
    embedding : Embedding
        An instance of the Embedding class for encoding text into embeddings.
    qa_model : BERTQuestionAnswer
        An instance of the BERTQuestionAnswer class for generating answers to questions.
    metrics : Metrics_Automated
        An instance of the Metrics_Automated class for evaluating the model's performance.
    k : int
        The number of dimensions for the KDTree (should match the embedding size).
    top_k : int
        The number of top context chunks to retrieve when answering questions.
    sentences_per_chunk : int
        The number of sentences per chunk when splitting documents.
    kd_tree : KDTree or None
        The KDTree used for fast retrieval of relevant context, initialized during precomputation.

    Methods
    -------
    __predict(question, context):
        Generates an answer to the question based on the given context using the BERT model.
    __save_embeddings(document_filename, chunk_number, embedding):
        Stores the embeddings in a numpy format.
    __precompute():
        Automatically precomputes the embeddings for all text chunks in the corpus and constructs a KDTree.
    search_context(question):
        Returns the nearest neighbors of a question, retrieving the most relevant sentence chunks.
    answer_question(question):
        Generates an answer to the question by retrieving the most relevant context and using the BERT model.
    evaluate_question(question, ground_truth):
        Evaluates the model's performance on a given question and ground truth answer, returning the score and match result.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2',
                 qa_model_name='bert-large-uncased-whole-word-masking-finetuned-squad',
                 k=384,
                 sentences_per_chunk=2,
                 top_k=3):
        """
        Initializes the Pipeline class with a specified model name for embeddings,
        a question-answering model, and the number of dimensions for KDTree.
        Automatically precomputes embeddings and constructs the KDTree.

        Parameters
        ----------
        model_name : str
            The name of the pre-trained model to be used for encoding.
        qa_model_name : str
            The name of the pre-trained BERT model to be used for question answering.
        k : int
            The number of dimensions for KDTree (should match the embedding size).
        sentences_per_chunk : int
            The number of sentences per chunk when splitting documents.
        top_k : int
            The number of top context chunks to retrieve when answering questions.
        """
        self.embedding = Embedding(model_name)
        self.qa_model = BERTQuestionAnswer(qa_model_name)
        self.metrics = Metrics_Automated()
        self.k = k
        self.top_k = top_k
        self.sentences_per_chunk = sentences_per_chunk
        self.kd_tree = None

        # Automatically precompute embeddings and construct KDTree on initialization
        self.__precompute()

    def __predict(self, question, context):
        """
        Generates an answer to the question based on the given context using the BERT model.

        Parameters
        ----------
        question : str
            The question to be answered.
        context : list
            A list of context strings from which to derive the answer.

        Returns
        -------
        str
            The generated answer to the question.
        """
        # Use the BERT question-answering model to generate an answer
        return self.qa_model.get_answer(question, context)

    def __save_embeddings(self, document_filename, chunk_number, embedding):
        """
        Store the embeddings in a numpy format.

        Parameters
        ----------
        document_filename : str
            The filename of the document.
        chunk_number : int
            The chunk number within the document.
        embedding : np.ndarray
            The embedding to be saved.
        """
        filename = f"{os.path.splitext(os.path.basename(document_filename))[0]}_{chunk_number}.npy"
        np.save(os.path.join('storage', 'embeddings', filename), embedding)

    def __precompute(self):
        """
        Automatically precompute the embeddings for all text chunks in storage/corpus/*.txt.clean 
        and construct a KDTree for fast retrieval.
        """
        # Ensure the embedding directory exists
        os.makedirs(os.path.join('storage', 'embeddings'), exist_ok=True)

        document_processor = DocumentProcessing()
        embeddings = []
        metadata_list = []

        # Process each document in the corpus
        documents = glob("storage/corpus/*.txt.clean")
        for document in documents:
            # Split document into chunks based on the specified number of sentences
            chunks = document_processor.split_document(document, sentences_per_chunk=self.sentences_per_chunk)
            for i, chunk in enumerate(chunks):
                # Encode each chunk into an embedding
                embedding = self.embedding.encode(chunk)
                # Save the embedding to disk
                self.__save_embeddings(document, i, embedding)
                embeddings.append(embedding)
                metadata_list.append({"document": document, "chunk": i, "text": chunk})

        # Construct KDTree for fast retrieval
        self.kd_tree = KDTree(k=self.k, points=embeddings, metadata_list=metadata_list)

    def precompute(self):        
        self.__precompute()
        return True

    def search_context(self, question):
        """
        Returns the nearest neighbors of a question. It retrieves the most relevant 
        sentence chunks by searching the KDTree.

        Parameters
        ----------
        question : str
            The question to search context for.

        Returns
        -------
        list
            The most relevant sentence chunks.
        """
        top_k = self.top_k
        # Encode the question into an embedding
        question_embedding = self.embedding.encode(question)
        # Search for the nearest neighbors in the KDTree
        searcher = KDTreeSearch(tree=self.kd_tree)
        nearest_neighbors = searcher.find_nearest_neighbors(point=question_embedding, k=top_k)
        return nearest_neighbors

    def answer_question(self, question):
        """
        Generates an answer to the question by retrieving the most relevant context
        and using the BERT model to derive the answer.

        Parameters
        ----------
        question : str
            The question to be answered.

        Returns
        -------
        str
            The generated answer to the question.
        """
        # Retrieve the most relevant context chunks for the question
        relevant_contexts = self.search_context(question)
        # Extract the text from each relevant context
        context_texts = [context[1]['text'] for context in relevant_contexts]

        # Generate the answer using the BERT question-answering model
        return self.__predict(question, context_texts)

    def evaluate_question(self, question, ground_truth):
        """
        Evaluates the model's performance on a given question and ground truth answer.
        Returns the specific score for the answer and a boolean indicating the match.

        Parameters
        ----------
        question : str
            The question to be evaluated.
        ground_truth : str
            The correct answer to the question.

        Returns
        -------
        tuple
            A tuple containing the ground truth, the model's answer, the specific score associated 
            with the answer, and a boolean indicating if the match is successful.
        """
        # Generate the answer using the pipeline
        model_answer = self.answer_question(question)

        # Calculate transformer match scores using the Metrics_Automated class
        transformer_scores, answer_match = self.metrics.transformer_match(model_answer, ground_truth, question)

        # Assuming we're interested in the score for the model's answer
        answer_score = transformer_scores[ground_truth][model_answer]
        return ground_truth, model_answer, answer_score, answer_match

if __name__ == "__main__":
    # Change to the correct directory
    os.chdir('retrieval_augmented_generation/')
    
    # Initialize the pipeline
    pipeline = Pipeline()

    # Example of evaluating a question
    question = "What is the capital of France?"
    ground_truth = "Paris"
    ground_truth, model_answer, score, match_result = pipeline.evaluate_question(question, ground_truth)
    print(f"Ground Truth: {ground_truth}")
    print(f"Model Answer: {model_answer}")
    print(f"Score: {score}")
    print(f"Match Result: {match_result}")
