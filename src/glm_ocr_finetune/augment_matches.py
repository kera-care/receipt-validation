from abc import ABC, abstractmethod
import csv
import json
from rapidfuzz import process as rf_process
from rapidfuzz.distance import JaroWinkler
from thefuzz import fuzz
from tqdm import tqdm
import numpy as np
from glm_ocr_finetune.data.utils import normalize_drug_name


class StringMatcher(ABC):
    """
    Abstract base class for string matching algorithms.
    """

    def __init__(self, strings: list[str], threshold: float = 0.8, top_n: int = 3):
        self.strings = [string.strip().lower() for string in strings]
        self.strings = list(set(self.strings))
        self.strings.sort()
        self.threshold = threshold
        self.top_n = top_n

    @abstractmethod
    def get_scores(self, query: str) -> np.ndarray:
        """
        Abstract method to compute match scores for a query.
        """
        pass

    def get_top_matches(self, query: str, return_scores: bool = False) -> list[str] | list[tuple[str, float]]:
        """
        Get the top matches for a given query based on match scores.

        Args:
            query (str): The string to be matched against the list of strings.
            return_scores (bool): Whether to return scores along with matches.
        Returns:
            list[str] | list[tuple[str, float]]: A list of top matching strings or 
            a list of tuples containing strings and their scores.
        """
        scores = self.get_scores(query)
        top_indices = np.argsort(scores)[::-1][: self.top_n]
        if return_scores:
            return [(self.strings[i], float(scores[i])) for i in top_indices if scores[i] >= self.threshold]
        return [self.strings[i] for i in top_indices if scores[i] >= self.threshold]


class FuzzyMatch(StringMatcher):
    """
    Fuzzy matching implementation using Levenshtein distance.
    """

    def get_scores(self, query: str) -> np.ndarray:
        query = query.strip().lower()
        return np.array([fuzz.ratio(query, string) / 100 for string in self.strings])


class JaroWinklerMatch(StringMatcher):
    """
    Jaro-Winkler matching implementation backed by rapidfuzz (C-accelerated).
    """

    def get_scores(self, query: str) -> np.ndarray:
        query = query.strip().lower()
        return np.array([
            JaroWinkler.normalized_similarity(query, s) for s in self.strings
        ])


class DrugRootMatcher:
    """
    Maps raw drug name strings (predictions or labels) to canonical drug root names
    using Jaro-Winkler similarity against all known drug variants.

    Uses rapidfuzz.process.extract for fast C-accelerated batch matching.
    """

    def __init__(self, drug_roots: dict, threshold: float = 0.8, top_n: int = 10):
        """
        Args:
            drug_roots (dict): Loaded drug_roots.json. Keys are canonical drug names;
                               values have 'roots' and 'variants' lists.
            threshold (float): Minimum similarity score to consider a label match.
            top_n (int): Maximum number of top matches to return for predictions.
        """
        self.threshold = threshold
        self.top_n = top_n

        # Build parallel lists: normalized variant → (drug_key, primary_root)
        self._variants: list[str] = []
        self._drug_keys: list[str] = []
        self._roots: list[str] = []

        for drug_key, info in drug_roots.items():
            primary_root = info["roots"][0] if info["roots"] else drug_key
            for variant in info["variants"]:
                self._variants.append(normalize_drug_name(variant))
                self._drug_keys.append(drug_key)
                self._roots.append(primary_root)

    def match_prediction(self, drug_name: str) -> list[dict]:
        """
        Return up to top_n distinct-root matches for a predicted drug name.
        No threshold is applied — always return the top_n best matches.
        Each result contains root, drug_key, matched_variant, and score.
        """
        query = normalize_drug_name(drug_name)
        # Fetch extra candidates to allow deduplication by root
        candidates = rf_process.extract(
            query,
            self._variants,
            scorer=JaroWinkler.normalized_similarity,
            limit=self.top_n * 5,
        )

        seen_roots: set[str] = set()
        results: list[dict] = []
        for variant, score, idx in candidates:
            root = self._roots[idx]
            if root in seen_roots:
                continue
            seen_roots.add(root)
            results.append({
                "root": root,
                "drug_key": self._drug_keys[idx],
                "matched_variant": variant,
                "score": round(score, 4),
            })
            if len(results) == self.top_n:
                break

        return results

    def match_label(self, drug_name: str) -> tuple[str, str] | None:
        """
        Return (root, drug_key) for a label drug name if the best Jaro-Winkler
        score >= threshold, otherwise return None (caller keeps the original label).
        """
        query = normalize_drug_name(drug_name)
        candidates = rf_process.extract(
            query,
            self._variants,
            scorer=JaroWinkler.normalized_similarity,
            limit=1,
        )
        if not candidates:
            return None
        _, best_score, best_idx = candidates[0]
        if best_score >= self.threshold:
            return self._roots[best_idx], self._drug_keys[best_idx]
        return None


class ExclusionMatcher:
    """
    Maps a root or drug name to its closest entry in drugs_exclusion.csv
    using Jaro-Winkler similarity, then returns the matched entry's
    is_exclusion flag.
    """

    def __init__(self, exclusion_csv_path: str, top_n: int = 10):
        """
        Args:
            exclusion_csv_path: Path to drugs_exclusion.csv.
            top_n: Number of top candidates to consider when finding the best
                   match for a query (higher = slower but more accurate).
        """
        self.top_n = top_n
        self._names: list[str] = []
        self._is_exclusion: list[bool] = []

        with open(exclusion_csv_path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self._names.append(normalize_drug_name(row["drug_name"]))
                self._is_exclusion.append(row["is_exclusion"].strip().lower() == "true")

    def match(self, name: str) -> dict:
        """
        Return the best-matching exclusion entry for *name*.

        Returns a dict with:
            matched_name  – normalized name from the CSV
            is_exclusion  – bool from the CSV
            score         – Jaro-Winkler similarity (0–1)
        """
        query = normalize_drug_name(name)
        candidates = rf_process.extract(
            query,
            self._names,
            scorer=JaroWinkler.normalized_similarity,
            limit=self.top_n,
        )
        if not candidates:
            return {"matched_name": None, "is_exclusion": False, "score": 0.0}
        best_name, best_score, best_idx = candidates[0]
        return {
            "matched_name": best_name,
            "is_exclusion": self._is_exclusion[best_idx],
            "score": round(float(best_score), 4),
        }


def augment_results(
    inference_results: list[dict],
    drug_roots: dict,
    exclusion_csv_path: str = "resources/drugs_exclusion.csv",
    threshold: float = 0.8,
    top_n_predictions: int = 10,
) -> list[dict]:
    """
    For each inference result, map predicted and label drug names to canonical roots.

    - Predictions: top `top_n_predictions` Jaro-Winkler matches (with scores) per drug name.
    - Labels: map to primary root if best score >= threshold, else keep original.
    - Each root is matched against the exclusion list; `is_exclusion` is added to
      every prediction top-match and to every label entry.

    Args:
        inference_results: Loaded inference_results.json items.
        drug_roots: Loaded drug_roots.json.
        exclusion_csv_path: Path to drugs_exclusion.csv.
        threshold: Minimum similarity score for label matching.
        top_n_predictions: Number of top matches to return per predicted drug name.

    Returns:
        List of augmented result dicts.
    """
    matcher = DrugRootMatcher(drug_roots, threshold=threshold, top_n=top_n_predictions)
    excl_matcher = ExclusionMatcher(exclusion_csv_path, top_n=10)

    augmented = []
    for item in tqdm(inference_results, desc="Augmenting results"):
        result = {
            "transaction_id": item["transaction_id"],
            "prescription_image_urls": item.get("prescription_image_urls", []),
        }

        # --- Labels (verified_drug_names) ---
        label_drug_names = item.get("verified_drug_names", [])
        mapped_labels = []
        for label in label_drug_names:
            match = matcher.match_label(label)
            if match is not None:
                root, drug_key = match
            else:
                root, drug_key = None, None
            mapped_root = root if root is not None else label
            excl = excl_matcher.match(mapped_root)
            mapped_labels.append({
                "original": label,
                "mapped_root": mapped_root,
                "drug_key": drug_key,
                "was_mapped": match is not None,
                "is_exclusion": excl["is_exclusion"],
                "exclusion_matched_name": excl["matched_name"],
                "exclusion_score": excl["score"],
            })
        result["labels"] = mapped_labels

        # --- Predictions ---
        predicted_drug_names = item.get("predictions", {}).get("drug_names", [])
        mapped_predictions = []
        for pred in predicted_drug_names:
            matches = matcher.match_prediction(pred)
            enriched_matches = []
            for m in matches:
                excl = excl_matcher.match(m["root"])
                enriched_matches.append({
                    **m,
                    "is_exclusion": excl["is_exclusion"],
                    "exclusion_matched_name": excl["matched_name"],
                    "exclusion_score": excl["score"],
                })
            mapped_predictions.append({
                "original": pred,
                "top_matches": enriched_matches,
            })
        result["predictions"] = mapped_predictions

        augmented.append(result)

    return augmented


def main():
    inference_path = "dev_drug_names_inference_results.json"
    drug_roots_path = "resources/drug_roots.json"
    output_path = "outputs/augmented_results.json"

    print(f"Loading inference results from {inference_path} ...")
    with open(inference_path, "r") as f:
        inference_results = json.load(f)
    print(f"  Loaded {len(inference_results)} records.")

    print(f"Loading drug roots from {drug_roots_path} ...")
    with open(drug_roots_path, "r") as f:
        drug_roots = json.load(f)
    print(f"  Loaded {len(drug_roots)} drug entries ({sum(len(v['variants']) for v in drug_roots.values())} total variants).")

    print("Augmenting results with Jaro-Winkler root matching ...")
    augmented = augment_results(
        inference_results=inference_results,
        drug_roots=drug_roots,
        exclusion_csv_path="resources/drugs_exclusion.csv",
        threshold=0.8,
        top_n_predictions=10,
    )

    print(f"Saving augmented results to {output_path} ...")
    with open(output_path, "w") as f:
        json.dump(augmented, f, indent=2, ensure_ascii=False)

    print("Done.")

    # Print a quick sample
    sample = augmented[1]
    print("\n--- Sample (record 1) ---")
    print(json.dumps(sample, indent=2))


if __name__ == "__main__":
    main()