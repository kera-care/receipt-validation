from collections.abc import Callable
from dataclasses import dataclass
import random

from glm_ocr_finetune.data.matchers import DEFAULT_SIMILARITY_THRESHOLD, StringMatcher

# OCR-based corruption patterns with weights
OCR_SUBSTITUTIONS = [
    # Common OCR confusions (character level) — lowercase only, uppercase never matches normalized names
    ('l', '1'), ('o', '0'), ('s', '5'), ('i', 'l'),
    ('cl', 'd'), ('rn', 'm'), ('vv', 'w'), ('nn', 'm'), ('li', 'h'),
    
    # Common pharmaceutical confusions
    ('ph', 'f'), ('th', 't'), ('tion', 'ton'),
    ('ine', 'me'), ('pine', 'pme'),
    
    # Space-related errors
    (' ', '')
]

# Common pharmaceutical suffixes/prefixes for generating plausible fake names
PHARMA_SUFFIXES = [
    'ine', 'ol', 'ide', 'ate', 'ene', 'one', 'cin', 'xin',
    'statin', 'pril', 'sartan', 'olol', 'azole', 'mycin'
]

PHARMA_PREFIXES = [
    'amo', 'met', 'pro', 'hydro', 'epi', 'anti', 'cef', 'azi',
    'levo', 'fluo', 'pred', 'dexa', 'chlor', 'sulfa'
]

def ocr_perturbation(name: str, max_attempts: int = 10) -> str:
    """ Generate a perturbed version of the drug name by applying common OCR confusions. """
    substitution = random.choice(OCR_SUBSTITUTIONS)
    perturbed = name.replace(substitution[0], substitution[1])
    attempt = 0
    while perturbed == name and attempt < max_attempts:  # Ensure that we actually make a change
        substitution = random.choice(OCR_SUBSTITUTIONS)
        perturbed = name.replace(substitution[0], substitution[1])
        attempt += 1
    return perturbed

def fake_name_perturbation(name: str) -> str:
    """ Generate a fake drug name by replacing the prefix and/or suffix with common pharmaceutical patterns. """
    prefix = random.choice(PHARMA_PREFIXES)
    suffix = random.choice(PHARMA_SUFFIXES)
    # Replace existing prefix/suffix if they exist, otherwise add new ones
    perturbed = name
    for p in PHARMA_PREFIXES:
        if perturbed.startswith(p):
            perturbed = perturbed.replace(p, prefix, 1)
            break
    else:
        perturbed = prefix + perturbed
    
    for s in PHARMA_SUFFIXES:
        if perturbed.endswith(s):
            perturbed = perturbed[:-len(s)] + suffix
            break
    else:
        perturbed = perturbed + suffix
    
    return perturbed

def fuzzy_match_perturbation(name: str, matcher: StringMatcher, threshold: float = DEFAULT_SIMILARITY_THRESHOLD) -> str | None:
    """ Generate a perturbed version of the drug name that is still similar to the original name based on the matcher. 
    
    Args:
        name (str): The original drug name to perturb.
        matcher (StringMatcher): The matcher to use for finding similar drug names.
        threshold (float): The similarity threshold for considering a drug name as similar.

    Returns:
        str | None: A similar drug name if found, otherwise None.
    
    """
    similar_drugs = matcher.get_similar_drugs(name, threshold=threshold, exclude_query=True)
    if similar_drugs:
        return random.choice(similar_drugs)
    else:
        return None
    
@dataclass
class Perturbation:
    type: str
    description: str
    function: Callable[[str], str]

    def apply(self, name: str) -> str:
        return self.function(name)

class PerturbationPipeline:
    def __init__(self, 
        candidate_generator: StringMatcher,
        apply_probability: float = 0.5,
        perturbation_probabilities: dict[str, float] | None = None
    ):
        self.apply_probability = apply_probability
        self.candidate_generator = candidate_generator

        self.ocr_perturbation = Perturbation(
            type="ocr",
            description="Apply common OCR confusions to the drug name",
            function=ocr_perturbation,
        )

        self.fake_name_perturbation = Perturbation(
            type="fake_name",
            description="Generate a fake drug name by replacing the prefix and/or suffix with common pharmaceutical patterns",
            function=fake_name_perturbation,
        )

        self.fuzzy_match_perturbation = Perturbation(
            type="replace_with_similar",
            description="Replace the drug name with a similar drug name based on the matcher",
            function=lambda name: fuzzy_match_perturbation(name, self.candidate_generator, threshold=DEFAULT_SIMILARITY_THRESHOLD),
        )

        valid_types = {p.type for p in [self.ocr_perturbation, self.fake_name_perturbation, self.fuzzy_match_perturbation]}

        if perturbation_probabilities is not None:
            unknown_keys = set(perturbation_probabilities) - valid_types
            if unknown_keys:
                raise ValueError(f"Unknown perturbation types: {unknown_keys}. Valid types: {valid_types}")
            probs = list(perturbation_probabilities.values())
            if not all(0 <= p <= 1 for p in probs):
                raise ValueError("All probabilities must be between 0 and 1.")
            total = sum(probs)
            if total == 0:
                raise ValueError("At least one probability must be greater than 0.")
            self.perturbation_probabilities = {k: v / total for k, v in perturbation_probabilities.items()}
        else:
            self.perturbation_probabilities = {
                "ocr": 0.4,
                "fake_name": 0.4,
                "replace_with_similar": 0.2
            }


    def perturb(self, drug_names: list[str], add_extra_probability: float = 0.1, remove_probability: float = 0.1) -> list[str]:
        if random.random() > self.apply_probability:
            return drug_names  # No perturbation applied
        
        output = []
        for name in drug_names:
            perturbation_type = random.choices(
                population=list(self.perturbation_probabilities.keys()),
                weights=list(self.perturbation_probabilities.values()),
                k=1
            )[0]

            if perturbation_type == "ocr":
                perturbed_name = self.ocr_perturbation.apply(name)
            elif perturbation_type == "fake_name":
                perturbed_name = self.fake_name_perturbation.apply(name)
            elif perturbation_type == "replace_with_similar":
                perturbed_name = self.fuzzy_match_perturbation.apply(name)
                if perturbed_name is None:  # Fallback to either OCR or fake name perturbation if no similar drug found
                    if random.random() < 0.5:
                        perturbed_name = self.ocr_perturbation.apply(name)
                    else:
                        perturbed_name = self.fake_name_perturbation.apply(name)
            else:
                # This should never happen due to the validation in __init__, but we include it for completeness
                raise ValueError(f"Unknown perturbation type: {perturbation_type}")
                

            output.append(perturbed_name)

        indices_to_remove = [i for i in range(len(output)) if random.random() < remove_probability]
        output = [name for i, name in enumerate(output) if i not in indices_to_remove]
        if random.random() < add_extra_probability:
            candidates = [name for name in self.candidate_generator.candidates if name not in output]
            if candidates:
                extra_name = random.choice(candidates)
                output.append(extra_name)

        return output


if __name__ == "__main__":
    # Example usage
    import structlog
    logger = structlog.get_logger(__name__)
    matcher = StringMatcher(candidates=["amoxicilline", "ibuprofene", "paracetamol", "amoxapine"])
    pipeline = PerturbationPipeline(candidate_generator=matcher, apply_probability=1.0)  # Always apply perturbation for testing
    original_names = ["amoxicilline", "ibuprofene"]
    perturbed_names = pipeline.perturb(original_names)
    logger.info("Original names", original=original_names)
    logger.info("Perturbed names", perturbed=perturbed_names)
