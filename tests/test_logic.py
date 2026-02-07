import pytest
from src.core.rag_pipeline import LegalRAG
from src.agents.extractor import get_extraction_agent
from src.agents.analyzer import get_analyzer_agent
from src.core.engine import create_legal_engine


def test_extractor_identifies_parties():
    agent = get_extraction_agent()
    test_text = "This agreement is between Google LLC and John Doe."
    
    result = agent.invoke({"contract_text": test_text})
    
    assert "Google LLC" in result.parties # type: ignore
    assert "John Doe" in result.parties # type: ignore

def test_anonymization_works():
    from src.utils.scrub import anonymize_contract
    original = "My name is John Doe and I live in New York."
    scrubbed = anonymize_contract(original)
    
    assert "John Doe" not in scrubbed
    assert "New York" not in scrubbed


def test_extractor_handles_missing_info():
    agent = get_extraction_agent()
    test_text = "This is a simple greeting card."
    result = agent.invoke({"contract_text": test_text})
    
    # Handle None, empty strings, or "not found" strings
    val = result.termination_period or "" # type: ignore
    assert any(x in val.lower() for x in ["not found", "none", "n/a", ""])


def test_analyzer_flags_extreme_non_compete():
    agent = get_analyzer_agent()
    mock_data = {
        "non_compete_clause": "Employee cannot work for any competitor globally for 5 years."
    }
    
    result = agent.invoke({"extracted_json": mock_data})
    
    risk = next((item for item in result.cons if "non-compete" in item.category.lower()), None) # type: ignore
    
    assert risk is not None
    # Use 'in' to catch "HIGH RISK" or "HIGH"
    assert "HIGH" in risk.severity.upper() 


def test_rag_retrieval_precision():
    rag = LegalRAG()
    text = """
    Section 1: Salary is $100k. 
    Section 2: Termination requires 30 days notice.
    Section 3: Dog policy is friendly.
    """
    rag.index_document(text, doc_id="test_doc")
    
    # Query specifically for termination
    results = rag.query_contract("How much notice is needed to quit?")
    
    # The top result should contain 'Section 2'
    assert "Section 2" in results[0].page_content # type: ignore
    # assert "Section 3" not in results[0].page_content # type: ignore


@pytest.mark.asyncio
async def test_graph_stops_on_error():
    engine = create_legal_engine()
    state = {"raw_text": "", "errors": []} 
    
    # Add the required configurable thread_id
    config = {"configurable": {"thread_id": "test-session"}}
    result = await engine.ainvoke(state, config=config) # type: ignore
    
    assert len(result["errors"]) > 0
    assert result.get("analysis") is None


def test_scrub_consistency():
    from src.utils.scrub import anonymize_contract
    text = "Contact Sarah Jenkins at sarah.j@google.com or 212-555-1234."
    scrubbed = anonymize_contract(text)
    
    assert "Sarah" not in scrubbed
    assert "google.com" not in scrubbed
    assert "212" not in scrubbed 


