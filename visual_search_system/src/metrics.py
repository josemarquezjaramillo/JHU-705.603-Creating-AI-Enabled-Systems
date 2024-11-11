import numpy as np

class RankingMetrics:
    def __init__(self, k):
        self.k = k

    def precision_at_k(self, r):
        """
        Calculate precision at k.

        Parameters:
        r (list): Binary relevance list for a query.

        Returns:
        float: Precision at k.
        """
        r = np.asarray(r)[:self.k]
        return np.mean(r)

    def recall_at_k(self, r, all_positives):
        """
        Calculate recall at k.

        Parameters:
        r (list): Binary relevance list for a query.
        all_positives (int): Total number of relevant items.

        Returns:
        float: Recall at k.
        """
        r = np.asarray(r)[:self.k]
        return np.sum(r) / all_positives

    def average_precision_at_k(self, r):
        """
        Calculate average precision at k.

        Parameters:
        r (list): Binary relevance list for a query.

        Returns:
        float: Average precision at k.
        """
        r = np.asarray(r)[:self.k]
        out = [self.precision_at_k(r) for i in range(1, self.k) if r[i]]
        if not out:
            return 0.0
        return np.mean(out)

    def mean_reciprocal_rank_at_k(self, rs):
        """
        Calculate mean reciprocal rank at k.

        Parameters:
        rs (list of lists): List of binary relevance lists for multiple queries.
        k (int): The number of top results to consider.

        Returns:
        float: Mean reciprocal rank at k.
        """
        return np.mean([1 / (np.argmax(r[:self.k]) + 1) if np.sum(r[:self.k]) > 0 else 0 for r in rs])


if __name__ == "__main__":
    relevance_ranks = [
        [0, 0, 1, 0, 1],   # The relevant documents are at ranks 3 and 5
        [1, 0, 0, 0, 0],   # The relevant document is at rank 1
        [0, 1, 0, 0, 0],   # The relevant document is at rank 2
        [0, 0, 0, 0, 0],   # No relevant document
    ]

    k = 3
    all_positives = [2, 1, 1, 0]  # Number of relevant documents for each query

    metrics = RankingMetrics(k=k)

    precision = [metrics.precision_at_k(r) for r in relevance_ranks]
    recall = [metrics.recall_at_k(r, all_pos) for r, all_pos in zip(relevance_ranks, all_positives)]
    ap = [metrics.average_precision_at_k(r) for r in relevance_ranks]
    mrr = metrics.mean_reciprocal_rank_at_k(relevance_ranks)

    print(f"Precision@{k}: {np.mean(precision):.4f}")
    print(f"Recall@{k}: {np.mean(recall):.4f}")
    print(f"MAP@{k}: {np.mean(ap):.4f}")
    print(f"MRR@{k}: {mrr:.4f}")
