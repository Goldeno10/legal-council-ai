import pytest
from src.agents.extractor import get_extraction_agent

def test_extractor_identifies_parties():
    agent = get_extraction_agent()
    test_text = "This agreement is between Google LLC and John Doe."
    
    result = agent.invoke({"contract_text": test_text})
    
    assert "Google LLC" in result.parties
    assert "John Doe" in result.parties

def test_anonymization_works():
    from src.utils.scrub import anonymize_contract
    original = "My name is John Doe and I live in New York."
    scrubbed = anonymize_contract(original)
    
    assert "John Doe" not in scrubbed
    assert "New York" not in scrubbed
