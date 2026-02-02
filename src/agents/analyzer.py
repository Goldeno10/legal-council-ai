from typing import List
from pydantic import BaseModel, Field
# from langchain_deepseek import ChatDeepSeek
from get_model import get_model
from langchain_core.prompts import ChatPromptTemplate

class RiskItem(BaseModel):
    category: str = Field(description="e.g., Non-compete, Liability, Termination")
    severity: str = Field(description="High, Medium, or Low")
    clause_reference: str = Field(description="The exact text or section number from the doc")
    explanation: str = Field(description="Why this is a risk for the user")
    suggestion: str = Field(description="How to renegotiate this clause")

class LegalAnalysis(BaseModel):
    pros: List[str] = Field(description="Positive aspects of the contract")
    cons: List[RiskItem] = Field(description="Detailed breakdown of risks")
    summary: str = Field(description="1-sentence plain English takeaway")

def get_analyzer_agent():
    # We use 'deepseek-reasoner' for its chain-of-thought capabilities
    # llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0)
    llm = get_model(temperature=0)
    
    # Define the "Playbook" - this is your system's "Opinion"
    playbook = """
    1. NON-COMPETE: Any non-compete over 6 months or covering a whole continent is HIGH RISK.
    2. TERMINATION: Notice periods longer than 3 months are MEDIUM RISK.
    3. IP OWNERSHIP: Ensure 'Moral Rights' are waived and work-for-hire is clearly defined.
    4. INDEMNITY: Employee should never indemnify the company for general business risks.
    """

    prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a Senior Legal Counsel. Analyze the extracted contract data against this Playbook:\n{playbook}"),
        ("user", "Extracted Data: {extracted_json}\n\nProvide a professional risk-benefit analysis.")
    ])
    
    return prompt | llm.with_structured_output(LegalAnalysis)
