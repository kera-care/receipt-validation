DRUG_NAME_EXTRACTION_PROMPTS = {
    "short": "Extract lexically sorted drug names in json format with the following format: {\"drug_names\": [\"drug_name1\", \"drug_name2\", ...]}",
}

PRESCRIPTION_VALIDATION_PROMPTS = {
    "short": """Given a prescription image extract the following information in JSON format:
- drug_names: A list of drug names mentioned in the prescription, sorted lexically.
- is_prescription: A boolean indicating whether the image is a prescription or not.
- has_stamp: A boolean indicating whether the prescription has a stamp or not.
- has_signature: A boolean indicating whether the prescription has a signature or not.
- date: The date mentioned in the prescription, if any, in ISO format (YYYY-MM-DD). If no date is mentioned, return null.
The output should be in the following JSON format:
{
    "drug_names": ["drug_name1", "drug_name2", ...],
    "is_prescription": true/false,
    "has_stamp": true/false,
    "has_signature": true/false,
    "date": "YYYY-MM-DD" or null
}
""",
}