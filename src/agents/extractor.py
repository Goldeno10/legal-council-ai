from pydantic import BaseModel, Field
from typing import List, Optional
from get_model import get_model
# from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate

# Define what we want to find in the contract
class LegalExtraction(BaseModel):
    parties: List[str] = Field(description="Names of the entities involved")
    termination_period: str = Field(description="Notice period for termination")
    non_compete_clause: Optional[str] = Field(description="Summary of non-compete restrictions")
    salary_and_benefits: str = Field(description="Summary of financial compensation")

def get_extraction_agent():
    """
    Extract specific data into a structured JSON format

    Example Execution:
    agent = get_extraction_agent()
    result = agent.invoke({"contract_text": md_content})
    print(result.non_compete_clause)
    """
    # llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0)
    llm = get_model(temperature=0)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert legal analyst. Extract the following details from the contract into a structured JSON format."),
        ("user", "Contract Content:\n{contract_text}")
    ])
    
    # Bind the schema to the LLM (Structured Output)
    return prompt | llm.with_structured_output(LegalExtraction)


