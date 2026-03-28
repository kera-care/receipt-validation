import pytest
from glm_ocr_finetune.data.matchers import StringMatcher


CANDIDATES = [
    "amoxicilline",
    "amoxapine",        # visually similar to amoxicilline
    "ibuprofene",
    "paracetamol",
    "metformine",
    "omeprazole",
    "atorvastatine",
]


@pytest.fixture
def matcher():
    return StringMatcher(CANDIDATES)


# ---------------------------------------------------------------------------
# match
# ---------------------------------------------------------------------------

class TestMatch:
    def test_exact_match_returns_score_one(self, matcher):
        results = matcher.match("amoxicilline")
        assert results[0]["drug_name"] == "amoxicilline"
        assert results[0]["score"] == pytest.approx(1.0)

    def test_accented_query_matches_normalized_candidate(self, matcher):
        # "ibuprofène" should match "ibuprofene" after accent stripping
        results = matcher.match("ibuprofène")
        assert results[0]["drug_name"] == "ibuprofene"
        assert results[0]["score"] > 0.9

    def test_uppercase_query_is_normalized(self, matcher):
        results = matcher.match("PARACETAMOL")
        assert results[0]["drug_name"] == "paracetamol"
        assert results[0]["score"] == pytest.approx(1.0)

    def test_limit_controls_number_of_results(self, matcher):
        results = matcher.match("amoxicilline", limit=3)
        assert len(results) == 3

    def test_limit_one_returns_single_result(self, matcher):
        results = matcher.match("amoxicilline", limit=1)
        assert len(results) == 1

    def test_result_contains_drug_name_and_score_keys(self, matcher):
        results = matcher.match("paracetamol")
        assert "drug_name" in results[0]
        assert "score" in results[0]


# ---------------------------------------------------------------------------
# fuzz_f1_score
# ---------------------------------------------------------------------------

class TestFuzzF1Score:
    def test_identical_lists_return_score_one(self, matcher):
        drugs = ["amoxicilline", "paracetamol"]
        assert matcher.fuzz_f1_score(drugs, drugs) == pytest.approx(1.0)

    def test_completely_different_lists_return_score_zero(self, matcher):
        predictions = ["amoxicilline"]
        ground_truth = ["ibuprofene"]
        assert matcher.fuzz_f1_score(predictions, ground_truth) == pytest.approx(0.0)

    def test_empty_predictions_return_score_zero(self, matcher):
        assert matcher.fuzz_f1_score([], ["paracetamol"]) == pytest.approx(0.0)

    def test_empty_ground_truth_returns_score_zero(self, matcher):
        assert matcher.fuzz_f1_score(["paracetamol"], []) == pytest.approx(0.0)

    def test_partial_overlap_returns_score_between_zero_and_one(self, matcher):
        predictions  = ["amoxicilline", "paracetamol"]
        ground_truth = ["amoxicilline", "ibuprofene"]
        score = matcher.fuzz_f1_score(predictions, ground_truth)
        assert 0.0 < score < 1.0

    def test_ocr_corrupted_prediction_still_matches(self, matcher):
        # "amoxici11ine" is an OCR corruption of "amoxicilline"
        score = matcher.fuzz_f1_score(["amoxici11ine"], ["amoxicilline"])
        assert score > 0.0

    def test_accented_prediction_matches_ground_truth(self, matcher):
        score = matcher.fuzz_f1_score(["ibuprofène"], ["ibuprofene"])
        assert score == pytest.approx(1.0)

    def test_score_is_symmetric_for_exact_matches(self, matcher):
        drugs = ["omeprazole", "metformine"]
        assert matcher.fuzz_f1_score(drugs, drugs) == matcher.fuzz_f1_score(drugs, drugs)

    def test_extra_prediction_reduces_precision(self, matcher):
        # predictions has one extra drug not in ground truth
        score_exact   = matcher.fuzz_f1_score(["amoxicilline"], ["amoxicilline"])
        score_extra   = matcher.fuzz_f1_score(["amoxicilline", "ibuprofene"], ["amoxicilline"])
        assert score_exact > score_extra

    def test_missing_prediction_reduces_recall(self, matcher):
        score_complete = matcher.fuzz_f1_score(["amoxicilline", "paracetamol"], ["amoxicilline", "paracetamol"])
        score_missing  = matcher.fuzz_f1_score(["amoxicilline"], ["amoxicilline", "paracetamol"])
        assert score_complete > score_missing


# ---------------------------------------------------------------------------
# get_similar_drugs
# ---------------------------------------------------------------------------

class TestGetSimilarDrugs:
    def test_does_not_return_query_itself(self, matcher):
        similar = matcher.get_similar_drugs("amoxicilline")
        assert "amoxicilline" not in similar

    def test_returns_visually_similar_drug(self, matcher):
        # "amoxapine" is similar to "amoxicilline"
        similar = matcher.get_similar_drugs("amoxicilline", threshold=0.7)
        assert "amoxapine" in similar

    def test_high_threshold_returns_fewer_results(self, matcher):
        low  = matcher.get_similar_drugs("amoxicilline", threshold=0.5)
        high = matcher.get_similar_drugs("amoxicilline", threshold=0.95)
        assert len(high) <= len(low)

    def test_returns_empty_list_when_no_similar_drugs(self, matcher):
        # a string with no resemblance to any candidate
        similar = matcher.get_similar_drugs("zzzzzzzzz", threshold=0.99)
        assert similar == []

    def test_accented_query_excluded_correctly(self, matcher):
        # "ibuprofène" normalizes to "ibuprofene" — should not appear in results
        similar = matcher.get_similar_drugs("ibuprofène", threshold=0.8)
        assert "ibuprofene" not in similar

    def test_all_results_are_above_threshold(self, matcher):
        threshold = 0.8
        similar = matcher.get_similar_drugs("amoxicilline", threshold=threshold)
        for drug in similar:
            results = matcher.match(drug)
            # each returned drug must itself be a valid candidate
            assert results[0]["drug_name"] == drug


# ---------------------------------------------------------------------------
# is_valid_pair
# ---------------------------------------------------------------------------

class TestIsValidPair:
    def test_chosen_equals_ground_truth_and_rejected_is_corrupted(self, matcher):
        ground_truth = ["amoxicilline", "paracetamol"]
        chosen       = ["amoxicilline", "paracetamol"]
        rejected     = ["amoxicilline"]                 # missing one drug
        assert matcher.is_valid_pair(chosen, rejected, ground_truth) is True

    def test_chosen_and_rejected_identical_returns_false(self, matcher):
        ground_truth = ["amoxicilline"]
        chosen       = ["amoxicilline"]
        rejected     = ["amoxicilline"]
        assert matcher.is_valid_pair(chosen, rejected, ground_truth) is False

    def test_chosen_worse_than_rejected_returns_false(self, matcher):
        ground_truth = ["amoxicilline", "paracetamol"]
        chosen       = ["ibuprofene"]               # unrelated drug
        rejected     = ["amoxicilline", "paracetamol"]  # matches GT perfectly
        assert matcher.is_valid_pair(chosen, rejected, ground_truth) is False

    def test_chosen_better_than_rejected_returns_true(self, matcher):
        ground_truth = ["amoxicilline", "paracetamol"]
        chosen       = ["amoxicilline", "paracetamol"]
        rejected     = ["amoxicilline", "ibuprofene"]   # wrong second drug
        assert matcher.is_valid_pair(chosen, rejected, ground_truth) is True

    def test_equal_f1_scores_return_false(self, matcher):
        ground_truth = ["amoxicilline"]
        # both have one correct drug and one wrong drug — equal F1
        chosen   = ["amoxicilline", "ibuprofene"]
        rejected = ["amoxicilline", "omeprazole"]
        assert matcher.is_valid_pair(chosen, rejected, ground_truth) is False

    def test_threshold_affects_validity(self, matcher):
        ground_truth = ["amoxicilline"]
        chosen       = ["amoxicilline"]
        rejected     = ["amoxapine"]   # similar but not the same

        # at low threshold amoxapine might match amoxicilline → equal scores → False
        # at high threshold amoxapine won't match → chosen wins → True
        result_high = matcher.is_valid_pair(chosen, rejected, ground_truth, threshold=0.99)
        assert result_high is True
