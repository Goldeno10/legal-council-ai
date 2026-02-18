import os
import re
from typing import List
from pydantic import BaseModel, Field
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage


class RiskItem(BaseModel):
    category: str = Field(description="e.g., Non-compete, IP Ownership, Liability")
    severity: str = Field(description="High, Medium, or Low")
    clause_reference: str = Field(description="The snippet or section found")
    explanation: str = Field(description="Human-like explanation of why this matters to the user's career")
    suggestion: str = Field(description="Specific professional advice for negotiation")


class LegalAnalysis(BaseModel):
    pros: List[str] = Field(default_factory=list, description="Positive aspects for the user")
    cons: List[RiskItem] = Field(default_factory=list, description="Detailed professional risks")
    verdict: str = Field(description="Senior Counsel's final recommendation: 'Sign', 'Negotiate', or 'Walk Away'")
    summary: str = Field(description="A empathetic 2-sentence takeaway for a layman")


def clean_json_text(text: str) -> str:
    text = re.sub(r"```json\s*|\s*```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

def get_analyzer_agent():
    parser = PydanticOutputParser(pydantic_object=LegalAnalysis)
    llm = get_model(temperature=0) # Low temperature for consistent legal logic
    
    playbook = """
    - NON-COMPETE: Restrictions > 6 months or broad geography are HIGH RISK.
    - TERMINATION: Notice > 3 months is MEDIUM RISK.
    - IP OWNERSHIP: Work-for-hire must be explicit; 'Moral Rights' must be protected.
    - INDEMNITY: Employee should NOT cover general company business risks.
    - VAGUENESS: Any undefined 'Discretionary' powers for the employer are HIGH RISK.
    """
    
    # Human-like System Prompt
    system_instruction = (
        f"You are a Senior Legal Partner. Analyze the discovered contract data against this Playbook:\n{playbook}\n\n"
        "Your tone should be professional yet protective of your client. "
        "Don't just list facts; provide counsel. "
        "Determine a 'verdict' based on the overall balance of the document. "
        "You MUST return ONLY a JSON object."
    )

    if os.getenv("USE_LOCAL_AI") == "true":
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", "Discovered Data: {extracted_json}\n\nFormat your response as JSON: {format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())

        def local_chain(input_data):
            raw_response = (prompt | llm).invoke(input_data)
            content = raw_response.content if isinstance(raw_response, BaseMessage) else str(raw_response)
            sanitized_json = clean_json_text(content) # type: ignore
            try:
                return parser.parse(sanitized_json)
            except Exception as e:
                return LegalAnalysis(
                    pros=["Could not process pros"],
                    cons=[],
                    verdict="Negotiate (due to parsing error)",
                    summary=f"Analysis failed: {str(e)}"
                )
        return local_chain

    else:
        # Cloud Logic (DeepSeek/OpenAI)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", "Extracted Data: {extracted_json}")
        ])
        return prompt | llm.with_structured_output(LegalAnalysis)





# from typing import List
# from pydantic import BaseModel, Field
# # from langchain_deepseek import ChatDeepSeek
# from .get_model import get_model
# from langchain_core.prompts import ChatPromptTemplate

# class RiskItem(BaseModel):
#     category: str = Field(description="e.g., Non-compete, Liability, Termination")
#     severity: str = Field(description="High, Medium, or Low")
#     clause_reference: str = Field(description="The exact text or section number from the doc")
#     explanation: str = Field(description="Why this is a risk for the user")
#     suggestion: str = Field(description="How to renegotiate this clause")


# class LegalAnalysis(BaseModel):
#     # Use default_factory for lists and default for strings
#     pros: List[str] = Field(default_factory=list, description="Positive aspects")
#     cons: List[RiskItem] = Field(default_factory=list, description="Detailed risks")
#     summary: str = Field(default="", description="1-sentence takeaway")

# def get_analyzer_agent():
#     llm = get_model(temperature=0)
    
#     playbook = """
#     1. NON-COMPETE: Any non-compete over 6 months or covering a whole continent is HIGH RISK.
#     2. TERMINATION: Notice periods longer than 3 months are MEDIUM RISK.
#     3. IP OWNERSHIP: Ensure 'Moral Rights' are waived and work-for-hire is clearly defined.
#     4. INDEMNITY: Employee should never indemnify the company for general business risks.
#     """
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", (
#             f"You are a Senior Legal Counsel. Analyze against this Playbook:\n{playbook}\n\n"
#             "STRICT RULES:\n"
#             "1. Output ONLY a flat JSON object.\n"
#             "2. Do NOT use a top-level 'analysis' key.\n"
#             "3. You MUST use these exact keys: 'pros', 'cons', 'summary'.\n"
#             "4. Each item in 'cons' must match: {{'category': ..., 'severity': ..., 'clause_reference': ..., 'explanation': ..., 'suggestion': ...}}.\n"
#             "5. If a playbook item isn't found, leave the list empty. Do not invent new keys."
#         )),
#         ("user", "Extracted Data: {extracted_json}")
#     ])


#     # Ensure the model is forced into JSON mode if the provider supports it
#     return prompt | llm.with_structured_output(LegalAnalysis)
