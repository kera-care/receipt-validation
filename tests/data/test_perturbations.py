import random
from unittest.mock import patch

import pytest

from glm_ocr_finetune.data.matchers import StringMatcher
from glm_ocr_finetune.data.perturbations import (
    PHARMA_PREFIXES,
    PHARMA_SUFFIXES,
    PerturbationPipeline,
    fake_name_perturbation,
    fuzzy_match_perturbation,
    ocr_perturbation,
)


CANDIDATES = ["amoxicilline", "amoxapine", "ibuprofene", "paracetamol", "metformine"]


@pytest.fixture
def matcher():
    return StringMatcher(CANDIDATES)


@pytest.fixture
def pipeline(matcher):
    return PerturbationPipeline(candidate_generator=matcher, apply_probability=1.0)


# ---------------------------------------------------------------------------
# ocr_perturbation
# ---------------------------------------------------------------------------

class TestOcrPerturbation:
    def test_returns_string(self):
        assert isinstance(ocr_perturbation("amoxicilline"), str)

    def test_changes_name_when_substitution_applies(self):
        # 'l' → '1' is in OCR_SUBSTITUTIONS and "amoxicilline" contains 'l'
        with patch("glm_ocr_finetune.data.perturbations.random.choice", return_value=('l', '1')):
            result = ocr_perturbation("amoxicilline")
        assert result != "amoxicilline"
        assert "1" in result

    def test_returns_original_when_no_substitution_applies(self):
        # space removal on a name with no spaces never changes it
        with patch("glm_ocr_finetune.data.perturbations.random.choice", return_value=(' ', '')):
            result = ocr_perturbation("amoxicilline", max_attempts=3)
        assert result == "amoxicilline"

    def test_substitution_is_applied_to_all_occurrences(self):
        # "amoxicilline" has two 'l's — both should be replaced
        with patch("glm_ocr_finetune.data.perturbations.random.choice", return_value=('l', '1')):
            result = ocr_perturbation("amoxicilline")
        assert result == "amoxici11ine"

    def test_space_removal_applies_to_multi_word_name(self):
        with patch("glm_ocr_finetune.data.perturbations.random.choice", return_value=(' ', '')):
            result = ocr_perturbation("co amoxiclav")
        assert result == "coamoxiclav"

    def test_uppercase_chars_are_not_substituted(self):
        # Uppercase substitutions were removed — 'I' should not be changed
        name = "ibuprofene"
        with patch("glm_ocr_finetune.data.perturbations.random.choice", return_value=('I', '1')):
            result = ocr_perturbation(name, max_attempts=1)
        assert result == name


# ---------------------------------------------------------------------------
# fake_name_perturbation
# ---------------------------------------------------------------------------

class TestFakeNamePerturbation:
    def test_returns_string(self):
        assert isinstance(fake_name_perturbation("amoxicilline"), str)

    def test_result_differs_from_input(self):
        # With a different prefix/suffix the result should change
        random.seed(42)
        result = fake_name_perturbation("ibuprofene")
        assert isinstance(result, str)

    def test_existing_prefix_is_replaced(self):
        # "amoxicilline" starts with "amo" — it should be replaced
        with patch("glm_ocr_finetune.data.perturbations.random.choice", side_effect=["met", "ol"]):
            result = fake_name_perturbation("amoxicilline")
        assert result.startswith("met")
        assert not result.startswith("amo")

    def test_prefix_prepended_when_no_existing_prefix_matches(self):
        with patch("glm_ocr_finetune.data.perturbations.random.choice", side_effect=["levo", "ol"]):
            result = fake_name_perturbation("xyz")
        assert result.startswith("levo")

    def test_existing_suffix_is_replaced(self):
        # "amoxicilline" ends with "ine" — it should be replaced
        with patch("glm_ocr_finetune.data.perturbations.random.choice", side_effect=["met", "ol"]):
            result = fake_name_perturbation("amoxicilline")
        assert result.endswith("ol")
        assert not result.endswith("ine")

    def test_suffix_appended_when_no_existing_suffix_matches(self):
        with patch("glm_ocr_finetune.data.perturbations.random.choice", side_effect=["levo", "azole"]):
            result = fake_name_perturbation("xyz")
        assert result.endswith("azole")

    def test_result_contains_pharma_prefix_or_suffix(self):
        random.seed(0)
        result = fake_name_perturbation("paracetamol")
        has_prefix = any(result.startswith(p) for p in PHARMA_PREFIXES)
        has_suffix = any(result.endswith(s) for s in PHARMA_SUFFIXES)
        assert has_prefix or has_suffix


# ---------------------------------------------------------------------------
# fuzzy_match_perturbation
# ---------------------------------------------------------------------------

class TestFuzzyMatchPerturbation:
    def test_returns_similar_drug_from_candidates(self, matcher):
        # "amoxicilline" and "amoxapine" are similar — should return amoxapine
        result = fuzzy_match_perturbation("amoxicilline", matcher, threshold=0.7)
        assert result in CANDIDATES

    def test_does_not_return_query_itself(self, matcher):
        result = fuzzy_match_perturbation("amoxicilline", matcher, threshold=0.7)
        assert result != "amoxicilline"

    def test_returns_none_when_no_similar_drug_found(self, matcher):
        result = fuzzy_match_perturbation("amoxicilline", matcher, threshold=0.9999)
        assert result is None

    def test_returns_none_for_unrecognized_name(self, matcher):
        result = fuzzy_match_perturbation("zzzzzzzzz", matcher, threshold=0.8)
        assert result is None


# ---------------------------------------------------------------------------
# PerturbationPipeline — __init__ validation
# ---------------------------------------------------------------------------

class TestPerturbationPipelineInit:
    def test_default_probabilities_sum_to_one(self, pipeline):
        total = sum(pipeline.perturbation_probabilities.values())
        assert total == pytest.approx(1.0)

    def test_custom_probabilities_are_normalized(self, matcher):
        # 0.6 + 0.6 = 1.2 — each value is valid (≤ 1) but they don't sum to 1
        pipeline = PerturbationPipeline(
            candidate_generator=matcher,
            perturbation_probabilities={"ocr": 0.6, "fake_name": 0.6},
        )
        assert sum(pipeline.perturbation_probabilities.values()) == pytest.approx(1.0)

    def test_partial_type_set_is_accepted(self, matcher):
        pipeline = PerturbationPipeline(
            candidate_generator=matcher,
            perturbation_probabilities={"ocr": 0.6, "fake_name": 0.4},
        )
        assert set(pipeline.perturbation_probabilities) == {"ocr", "fake_name"}

    def test_unknown_perturbation_type_raises(self, matcher):
        with pytest.raises(ValueError, match="Unknown perturbation types"):
            PerturbationPipeline(
                candidate_generator=matcher,
                perturbation_probabilities={"ocr": 0.5, "typo": 0.5},
            )

    def test_all_zero_probabilities_raises(self, matcher):
        with pytest.raises(ValueError, match="At least one probability"):
            PerturbationPipeline(
                candidate_generator=matcher,
                perturbation_probabilities={"ocr": 0.0, "fake_name": 0.0},
            )

    def test_probability_out_of_range_raises(self, matcher):
        with pytest.raises(ValueError, match="between 0 and 1"):
            PerturbationPipeline(
                candidate_generator=matcher,
                perturbation_probabilities={"ocr": 1.5},
            )

    def test_negative_probability_raises(self, matcher):
        with pytest.raises(ValueError, match="between 0 and 1"):
            PerturbationPipeline(
                candidate_generator=matcher,
                perturbation_probabilities={"ocr": -0.1, "fake_name": 1.1},
            )


# ---------------------------------------------------------------------------
# PerturbationPipeline — perturb
# ---------------------------------------------------------------------------

class TestPerturbationPipelinePerturb:
    def test_returns_list_of_strings(self, pipeline):
        result = pipeline.perturb(["amoxicilline", "paracetamol"], add_extra_probability=0.0, remove_probability=0.0)
        assert isinstance(result, list)
        assert all(isinstance(name, str) for name in result)

    def test_apply_probability_zero_returns_original(self, matcher):
        pipeline = PerturbationPipeline(candidate_generator=matcher, apply_probability=0.0)
        drugs = ["amoxicilline", "paracetamol"]
        assert pipeline.perturb(drugs) == drugs

    def test_remove_probability_one_empties_list(self, pipeline):
        result = pipeline.perturb(["amoxicilline", "paracetamol"], add_extra_probability=0.0, remove_probability=1.0)
        assert result == []

    def test_remove_probability_zero_keeps_all(self, pipeline):
        drugs = ["amoxicilline", "paracetamol"]
        result = pipeline.perturb(drugs, add_extra_probability=0.0, remove_probability=0.0)
        assert len(result) == len(drugs)

    def test_add_extra_probability_one_adds_a_drug(self, pipeline):
        drugs = ["amoxicilline"]
        result = pipeline.perturb(drugs, add_extra_probability=1.0, remove_probability=0.0)
        assert len(result) == len(drugs) + 1

    def test_add_extra_probability_zero_does_not_add(self, pipeline):
        drugs = ["amoxicilline", "paracetamol"]
        result = pipeline.perturb(drugs, add_extra_probability=0.0, remove_probability=0.0)
        assert len(result) == len(drugs)

    def test_extra_drug_is_not_duplicate(self, pipeline):
        # Use all candidates except one so there is always a non-duplicate available
        drugs = ["amoxicilline", "amoxapine", "ibuprofene", "paracetamol"]
        result = pipeline.perturb(drugs, add_extra_probability=1.0, remove_probability=0.0)
        assert len(result) == len(set(result))

    def test_no_extra_added_when_all_candidates_already_present(self, matcher):
        # Disable perturbations so names stay unchanged, then all candidates are in output
        pipeline = PerturbationPipeline(candidate_generator=matcher, apply_probability=0.0)
        drugs = list(matcher.candidates)
        result = pipeline.perturb(drugs, add_extra_probability=1.0, remove_probability=0.0)
        assert len(result) == len(drugs)

    def test_output_length_with_no_add_or_remove(self, pipeline):
        drugs = ["amoxicilline", "paracetamol", "metformine"]
        result = pipeline.perturb(drugs, add_extra_probability=0.0, remove_probability=0.0)
        assert len(result) == len(drugs)
