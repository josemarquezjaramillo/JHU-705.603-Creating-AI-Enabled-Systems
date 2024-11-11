from qa_metrics.transformerMatcher import TransformerMatcher
from qa_metrics.em import em_match


class Metrics_Automated:

    def __init__(self, model="distilbert"):
        # Supported models: roberta-large, tiny-bert, roberta, bert, distilbert, distilroberta
        self.transformer_matcher = TransformerMatcher(model)


    def exact_match(self, generated_answer, ground_truth_answer):
        results = em_match(ground_truth_answer, generated_answer)
        return results
    
    def transformer_match(self, generated_answer, ground_truth_answer, question):
        scores = self.transformer_matcher.get_scores(ground_truth_answer, generated_answer, question)
        match_result = self.transformer_matcher.transformer_match(ground_truth_answer, generated_answer, question)
        return scores, match_result



if __name__ == "__main__":
    metrics = Metrics_Automated()

    """Question: What is the capital of France?"""
    
    question = "What is the capital of France?"
    # Correct answer example
    generated_answer = "Paris"
    ground_truth_answer = "Paris"
    
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')

    # Correct answer example
    generated_answer = "Paris"
    ground_truth_answer = "The capital of France is Paris."

    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')


    # Incorrect answer example
    generated_answer = "The capital of France is London, which is a city across the channel from Paris"
    ground_truth_answer = "Paris"
    
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')

    # Incorrect answer example
    generated_answer = "Nice, France"
    ground_truth_answer = "London"
    
    score = metrics.exact_match(generated_answer, ground_truth_answer)
    print(f'Exact Match: {score}')
    score = metrics.transformer_match(generated_answer, ground_truth_answer, question)
    print(f'Transformer Match: {score}')