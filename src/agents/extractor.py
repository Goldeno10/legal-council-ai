from pydantic import BaseModel, Field
from typing import List, Optional
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate

# Define what we want to find in the contract
class LegalExtraction(BaseModel):
    # Use default values so Pydantic doesn't crash on missing keys
    parties: List[str] = Field(default_factory=list, description="Names of the entities involved")
    termination_period: Optional[str] = Field(default=None, description="Notice period for termination")
    non_compete_clause: Optional[str] = Field(default=None, description="Summary of non-compete restrictions")
    salary_and_benefits: Optional[str] = Field(default=None, description="Summary of financial compensation")

def get_extraction_agent():
    """
    Extract specific data into a structured JSON format.
    """
    # temperature=0 is critical for consistent JSON from local models
    llm = get_model(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert legal analyst. "
            "Extract the following details from the contract. "
            "Return ONLY the requested fields in a flat JSON structure. "
            "Do not wrap the response in keys like 'contract_details' or 'extracted_details'. "
            "For parties, provide a simple list of strings containing only their names."
        )),
        ("user", "Contract Content:\n{contract_text}")
    ])
    
    # .with_structured_output is the most robust way to handle this
    # reference: https://python.langchain.com
    return prompt | llm.with_structured_output(LegalExtraction)
