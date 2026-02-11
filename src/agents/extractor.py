import os
import re
import json
from pydantic import BaseModel, Field
from typing import List, Optional
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage

# 1. Define the Schema
class LegalExtraction(BaseModel):
    parties: List[str] = Field(default_factory=list, description="Names of the entities involved")
    termination_period: Optional[str] = Field(default=None, description="Notice period for termination")
    non_compete_clause: Optional[str] = Field(default=None, description="Summary of non-compete restrictions")
    salary_and_benefits: Optional[str] = Field(default=None, description="Summary of financial compensation")

def clean_json_text(text: str) -> str:
    """
    Staff-level utility to strip Markdown markers and conversational filler.
    Ensures the string passed to the parser is strictly the JSON block.
    """
    # Remove markdown code blocks if present
    text = re.sub(r"```json\s*|\s*```", "", text)
    # Find the first '{' and last '}' to isolate the JSON object
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

def get_extraction_agent():
    parser = PydanticOutputParser(pydantic_object=LegalExtraction)
    llm = get_model(temperature=0)

    if os.getenv("USE_LOCAL_AI") == "true":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal data extractor. You must return ONLY JSON. No conversation or markdown."),
            ("user", "Extract data from this text: {contract_text}\n\nReturn JSON matching this schema: {format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())

        # Logic for Local: LLM -> Clean Text -> Parse into Pydantic
        def local_chain(input_data):
            # 1. Generate Raw Response
            raw_response = (prompt | llm).invoke(input_data)
            content = raw_response.content if isinstance(raw_response, BaseMessage) else str(raw_response)
            
            # 2. Clean the string (Crucial for Llama/DeepSeek local)
            sanitized_json = clean_json_text(content) # type: ignore
            
            # 3. Parse into Pydantic model
            try:
                return parser.parse(sanitized_json)
            except Exception as e:
                # Senior fallback: return partially valid model instead of crashing the stream
                return LegalExtraction(parties=["Parsing Error"], termination_period=f"Error: {str(e)}")

        return local_chain

    else:
        # Cloud Logic: Uses native tool-calling/structured output
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert legal analyst. Return a flat JSON structure."),
            ("user", "Contract Content:\n{contract_text}")
        ])
        return prompt | llm.with_structured_output(LegalExtraction)



# import os
# from pydantic import BaseModel, Field
# from typing import List, Optional
# from .get_model import get_model
# from langchain_core.prompts import ChatPromptTemplate


# from langchain_core.output_parsers import PydanticOutputParser



# # Define what we want to find in the contract
# class LegalExtraction(BaseModel):
#     # Use default values so Pydantic doesn't crash on missing keys
#     parties: List[str] = Field(default_factory=list, description="Names of the entities involved")
#     termination_period: Optional[str] = Field(default=None, description="Notice period for termination")
#     non_compete_clause: Optional[str] = Field(default=None, description="Summary of non-compete restrictions")
#     salary_and_benefits: Optional[str] = Field(default=None, description="Summary of financial compensation")

# def get_extraction_agent():
#     """
#     Extract specific data into a structured JSON format.
#     """
#     parser = PydanticOutputParser(pydantic_object=LegalExtraction)

#     # temperature=0 is critical for consistent JSON from local models
#     llm = get_model(temperature=0)

#     if os.getenv("USE_LOCAL_AI") == "true":
        
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", "You are an expert legal analyst. You MUST respond ONLY with a valid JSON object."),
#             ("user", "Extract details from this contract:\n{contract_text}\n\nFormat your response according to this schema: {format_instructions}")
#         ]).partial(format_instructions=parser.get_format_instructions())
#     else:
#         prompt = ChatPromptTemplate.from_messages([
#             ("system", (
#                 "You are an expert legal analyst. "
#                 "Extract the following details from the contract. "
#                 "Return ONLY the requested fields in a flat JSON structure. "
#                 "Do not wrap the response in keys like 'contract_details' or 'extracted_details'. "
#                 "For parties, provide a simple list of strings containing only their names."
#             )),
#             ("user", "Contract Content:\n{contract_text}")
#         ])
    
#      # For local models, we pipe through the parser manually for better error handling
#     if os.getenv("USE_LOCAL_AI") == "true":
#         return prompt | llm | parser
#     else:
#         return prompt | llm.with_structured_output(LegalExtraction)
#     # .with_structured_output is the most robust way to handle this
#     # reference: https://python.langchain.com
#     # return prompt | llm.with_structured_output(LegalExtraction)
