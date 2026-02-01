import re


def verify_grounding(raw_text: str, clause_reference: str) -> bool:
    """
    Search for the identified clause in the raw text to ensure 
    the AI didn't 'hallucinate' a legal provision.
    """
    # Simple semantic or fuzzy match check
    # In a full-scale app, use a Cross-Encoder model here
    return clause_reference.lower() in raw_text.lower()
