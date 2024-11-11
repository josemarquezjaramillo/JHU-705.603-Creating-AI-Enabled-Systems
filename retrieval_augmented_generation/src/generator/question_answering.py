import torch
from transformers import BertTokenizer, BertForQuestionAnswering

class BERTQuestionAnswer:
    """
    A class used to perform question answering using a pre-trained BERT model.

    Methods
    -------
    get_answer(question: str, context: list) -> str
        Finds the answer to the question based on the given context.
    """

    def __init__(self, qa_model_name):
        """
        Initializes the BERTQuestionAnswer class with the specified model directory.

        Parameters
        ----------
        qa_model_name : str
            The name of the pre-trained BERT model to be used for question answering.
        """
        self.tokenizer = BertTokenizer.from_pretrained(qa_model_name)
        self.model = BertForQuestionAnswering.from_pretrained(qa_model_name)
        
    def get_answer(self, question, context):
        """
        Finds the answer to the question based on the given context.

        Parameters
        ----------
        question : str
            The question to be answered.
        context : list
            The context in which to find the answer, provided as a list of strings.

        Returns
        -------
        str
            The answer to the question.
        """
        # Join the context list into a single string
        context = "[SEP] ".join(context)

        # Tokenize the input question and context, ensuring truncation if too long
        inputs = self.tokenizer.encode_plus(
            question, 
            context, 
            max_length=512, 
            truncation=True, 
            return_tensors='pt'
        )
        
        # Get the start and end scores for the answer
        with torch.no_grad():
            outputs = self.model(**inputs)

        start_scores = outputs.start_logits
        end_scores = outputs.end_logits

        # Get the most likely beginning and end of the answer span
        start_index = torch.argmax(start_scores)
        end_index = torch.argmax(end_scores)

        # Decode the answer
        answer = self.tokenizer.convert_tokens_to_string(
            self.tokenizer.convert_ids_to_tokens(
                inputs['input_ids'][0][start_index:end_index+1]
            )
        )

        return answer

if __name__ == "__main__":
    qa_model_name = "bert-large-uncased-whole-word-masking-finetuned-squad"
    qa_model = BERTQuestionAnswer(qa_model_name)

    question = "What is the capital of France?"
    context = [
        "France, officially the French Republic, is a country primarily located in Western Europe.",
        "The capital of France is Paris."
    ]

    print(qa_model.get_answer(question, context))
