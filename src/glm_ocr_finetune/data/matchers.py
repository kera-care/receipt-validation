from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler
from glm_ocr_finetune.data.utils import normalize_drug_name


DEFAULT_SIMILARITY_THRESHOLD = 0.5


class StringMatcher:
    """
    A simple string matcher that uses Jaro-Winkler similarity to find top matches for a query string against a list of candidate strings.
        Returns a list of matches with their similarity scores.
    """

    def __init__(self, candidates: list[str]):
        """
        Initialize the matcher with a list of candidate strings.

        Args:
            candidates (list[str]): The list of strings to match against.
        """
        self.candidates = [normalize_drug_name(c) for c in candidates]

    def match(self, query: str, limit: int = 1) -> list[dict]:
        """
        Match the query string against the candidates and return the best match.

        Args:
            query (str): The string to match.
            limit (int): The maximum number of matches to return.
        Returns:
            list[dict]: A list of matches, each containing the matched string and its similarity score.

        """
        normalized_query = normalize_drug_name(query)
        matches = rf_process.extract(
            normalized_query,
            self.candidates,
            scorer=JaroWinkler.normalized_similarity,
            limit=limit,
        )
        return [{"drug_name": match[0], "score": match[1]} for match in matches]
    

    def fuzz_f1_score(
        self, predictions: list[str], ground_truths: list[str],
        prediction_threshold: float = DEFAULT_SIMILARITY_THRESHOLD, 
        reference_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        prediction_top_n: int = 1
    ) -> float:
        """
        Compute the F1 score between the predicted strings and the ground truth strings based on fuzzy matching.

        Args:
            predictions (list[str]): The list of predicted strings.
            ground_truths (list[str]): The list of ground truth strings.
            prediction_threshold (float): The minimum similarity score for a prediction to be considered a match.
            reference_threshold (float): The minimum similarity score for a reference to be considered a match.
            prediction_top_n (int): The number of top matches to consider for each prediction.

        Returns:
            float: The F1 score between the predictions and ground_truths.

        """

        matched_predictions = set()
        for pred in predictions:
            match = self.match(pred, limit=prediction_top_n)
            for m in match:
                if m["score"] > prediction_threshold:  # Consider a match if the score is above a threshold
                    matched_predictions.add(m["drug_name"])
        matched_ground_truths = set()
        for ref in ground_truths:
            match = self.match(ref, limit=1)
            if match and match[0]["score"] > reference_threshold:  # Consider a match if the score is above a threshold
                matched_ground_truths.add(match[0]["drug_name"])


        true_positives = len(matched_predictions.intersection(matched_ground_truths))
        precision = true_positives / len(matched_predictions) if matched_predictions else 0
        recall = true_positives / len(matched_ground_truths) if matched_ground_truths else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
    

    def get_similar_drugs(self, query: str, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> list[str]:
        """
        Get a list of drugs that are similar to the query string based on a similarity threshold.

        Args:
            query (str): The string to match.
            threshold (float): The minimum similarity score for a drug to be considered similar.

        Returns:
            list[str]: A list of similar drug names.
        """
        matches = self.match(query, limit=len(self.candidates))
        normalized_query = normalize_drug_name(query)
        similar_drugs = [
            match["drug_name"] 
            for match in matches if match["score"] > threshold and match["drug_name"] != normalized_query
        ]
        return similar_drugs
    

    def is_valid_pair(
        self, chosen: list[str], reference: list[str],
        ground_truth: list[str], threshold: float = DEFAULT_SIMILARITY_THRESHOLD
    ) -> bool:
        """
        Check if the chosen drug names are a valid match to the reference drug names based on their
        similarity to the ground truth drug names.

        The order of the drug names does not matter.

        Args:
            chosen (list[str]): The list of chosen drug names.
            reference (list[str]): The list of reference drug names.
            ground_truth (list[str]): The list of ground truth drug names.
            threshold (float): The minimum similarity score for a pair to be considered valid.

        Returns:
            bool: True if there is a valid pair, False otherwise. Valid means that the chosen drug names
            have a higher F1 score against the ground truth than the reference drug names do against
            the ground truth.
        """

        chosen_f1_score = self.fuzz_f1_score(
            chosen, ground_truth, prediction_threshold=threshold,
            reference_threshold=threshold, prediction_top_n=1
        )
        reference_f1_score = self.fuzz_f1_score(
            reference, ground_truth, prediction_threshold=threshold,
            reference_threshold=threshold, prediction_top_n=1
        )
        return chosen_f1_score > reference_f1_score
