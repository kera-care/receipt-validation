RECEIPT_VALIDATION_PROMPTS = {
    "short": """Given a document image, extract the following information in JSON format:
- is_health_receipt: A boolean indicating whether the image is a receipt (reçu, devis, commande, facture acquittée, ticket de caisse).
- total_amount: The total amount as a string (e.g. "12500", "3500.00"), or null if not found. Do not include the currency symbol.
- date: The date in ISO format (YYYY-MM-DD), or null.
- patient_name: The patient or customer name (nom du patient/client), or null.
- provider_info: The provider or merchant name and address (e.g. pharmacie, clinique, laboratoire, hôpital), or null.
- proof_of_payment: Description of payment evidence or an official stamp or signature from the provider (e.g. "cachet PAYÉ", "tampon REÇU", "signature du caissier", "PAYÉ stamp", "ACQUITTÉ", "paid stamp"), or null.
The output should be in the following JSON format:
{
    "is_health_receipt": true/false,
    "total_amount": "..." or null,
    "date": "YYYY-MM-DD" or null,
    "patient_name": "..." or null,
    "provider_info": "..." or null,
    "proof_of_payment": "..." or null
}
""",
}