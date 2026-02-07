from typing import List
from pydantic import BaseModel, Field
# from langchain_deepseek import ChatDeepSeek
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate

class RiskItem(BaseModel):
    category: str = Field(description="e.g., Non-compete, Liability, Termination")
    severity: str = Field(description="High, Medium, or Low")
    clause_reference: str = Field(description="The exact text or section number from the doc")
    explanation: str = Field(description="Why this is a risk for the user")
    suggestion: str = Field(description="How to renegotiate this clause")


class LegalAnalysis(BaseModel):
    # Use default_factory for lists and default for strings
    pros: List[str] = Field(default_factory=list, description="Positive aspects")
    cons: List[RiskItem] = Field(default_factory=list, description="Detailed risks")
    summary: str = Field(default="", description="1-sentence takeaway")


# class LegalAnalysis(BaseModel):
#     pros: List[str] = Field(description="Positive aspects of the contract")
#     cons: List[RiskItem] = Field(description="Detailed breakdown of risks")
#     summary: str = Field(description="1-sentence plain English takeaway")

def get_analyzer_agent():
    llm = get_model(temperature=0)
    
    playbook = """
    1. NON-COMPETE: Any non-compete over 6 months or covering a whole continent is HIGH RISK.
    2. TERMINATION: Notice periods longer than 3 months are MEDIUM RISK.
    3. IP OWNERSHIP: Ensure 'Moral Rights' are waived and work-for-hire is clearly defined.
    4. INDEMNITY: Employee should never indemnify the company for general business risks.
    """

    # prompt = ChatPromptTemplate.from_messages([
    #     ("system", (
    #         "You are a Senior Legal Counsel. Analyze the extracted contract data against this Playbook:\n"
    #         f"{playbook}\n\n"
    #         "CRITICAL: You must return ONLY a valid JSON object matching the provided schema. "
    #         "Do not include any conversational text, headers, or markdown formatting like '```json'. "
    #         "Your response must start with '{{' and end with '}}'."
    #     )),
    #     ("user", "Extracted Data: {extracted_json}")
    # ])
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            f"You are a Senior Legal Counsel. Analyze against this Playbook:\n{playbook}\n\n"
            "STRICT RULES:\n"
            "1. Output ONLY a flat JSON object.\n"
            "2. Do NOT use a top-level 'analysis' key.\n"
            "3. You MUST use these exact keys: 'pros', 'cons', 'summary'.\n"
            "4. Each item in 'cons' must match: {{'category': ..., 'severity': ..., 'clause_reference': ..., 'explanation': ..., 'suggestion': ...}}.\n"
            "5. If a playbook item isn't found, leave the list empty. Do not invent new keys."
        )),
        ("user", "Extracted Data: {extracted_json}")
    ])


    # Ensure the model is forced into JSON mode if the provider supports it
    return prompt | llm.with_structured_output(LegalAnalysis)



# def get_analyzer_agent():
#     # We use 'deepseek-reasoner' for its chain-of-thought capabilities
#     # llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0)
#     llm = get_model(temperature=0)
    
#     # Define the "Playbook" - this is your system's "Opinion"
#     playbook = """
#     1. NON-COMPETE: Any non-compete over 6 months or covering a whole continent is HIGH RISK.
#     2. TERMINATION: Notice periods longer than 3 months are MEDIUM RISK.
#     3. IP OWNERSHIP: Ensure 'Moral Rights' are waived and work-for-hire is clearly defined.
#     4. INDEMNITY: Employee should never indemnify the company for general business risks.
#     """

#     prompt = ChatPromptTemplate.from_messages([
#         ("system", f"You are a Senior Legal Counsel. Analyze the extracted contract data against this Playbook:\n{playbook}"),
#         ("user", "Extracted Data: {extracted_json}\n\nProvide a professional risk-benefit analysis.")
#     ])
    
#     return prompt | llm.with_structured_output(LegalAnalysis)
