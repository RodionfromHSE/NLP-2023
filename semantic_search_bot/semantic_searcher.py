import torch
from sentence_transformers import SentenceTransformer, util
from typing import List, Tuple

class SemanticSearcher:
    """
    SemanticSearcher is a class for performing semantic search using SentenceTransformer models.

    Args:
        model_name (str): The name of the SentenceTransformer model to use.

    Attributes:
        model: The SentenceTransformer model for embedding and similarity calculations.

    Methods:
        get_score(text_a: str, text_b: str) -> float:
            Calculate the adjusted similarity score between two text inputs.

        call(query: str, messages: List[str], n: int = 5) -> List[str]:
            Perform semantic search and return the top n most semantically close messages to the query.
    """

    def __init__(self, model_name: str = 'paraphrase-MiniLM-L6-v2'):
        """
        Initialize a SemanticSearcher instance with a specified SentenceTransformer model.
        """
        self.model = SentenceTransformer(model_name)

    def get_score(self, text_a: str, text_b: str) -> float:
        """
        Calculate the adjusted similarity score between two text inputs.

        Args:
            text_a (str): The first text input.
            text_b (str): The second text input.

        Returns:
            float: The adjusted similarity score in the range [0, 1], where 1 represents very close similarity.
        """
        embeddings_a = self.model.encode(text_a, convert_to_tensor=True)
        embeddings_b = self.model.encode(text_b, convert_to_tensor=True)
        similarity_score = util.pytorch_cos_sim(embeddings_a, embeddings_b)[0][0]
        # Adjust the similarity score to the range [0, 1] by mapping [-1, 1] to [0, 1]
        adjusted_score = 0.5 * (similarity_score + 1)
        return adjusted_score.item()

    def __call__(self, query: str, messages: List[str], n: int = 5, return_scores: bool = False) -> Tuple[List[str], List[float]] | List[str]:
        """
        Perform semantic search and return the top n most semantically close messages to the query.

        Args:
            query (str): The query text for which semantically close messages are sought.
            messages (List[str]): A list of text messages to compare against the query.
            n (int, optional): The number of top matching messages to return. Defaults to 5.
            return_scores (bool, optional): Whether to return the similarity scores of the top n matching messages. Defaults to False.

        Returns:
            List[str]: A list of the top n most semantically close messages to the query with their scores.
        """
        scores = [(msg, self.get_score(query, msg)) for msg in messages]
        sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[:n]
        top_n_messages, top_n_scores = zip(*sorted_scores)
        if return_scores:
            return top_n_messages, top_n_scores
        return top_n_messages
