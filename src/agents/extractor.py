import os
import re
import json
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage


class LegalDiscovery(BaseModel):
    is_legal_document: bool = Field(description="Is this actually a legal document?")
    document_type: str = Field(description="The specific type of agreement identified")
    parties: List[str] = Field(default_factory=list, description="Entities involved")
    complex_terms: List[Dict[str, str]] = Field(
        description="List of jargon terms (e.g., 'Indemnification') and their simple layman definitions"
    )
    key_obligations: List[str] = Field(description="What the user is actually required to do")
    hidden_risks: List[str] = Field(description="Subtle traps found in the fine print")


def clean_json_text(text: str) -> str:
    """
    Finds the outermost { } block. 
    Prevents crashes from 'Here is the JSON:' conversational filler.
    """
    try:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        return match.group(0) if match else text
    except:
        return text

def get_discovery_agent():
    """
    Discovery agent that replaces the rigid extractor with a cognitive one.
    """
    parser = PydanticOutputParser(pydantic_object=LegalDiscovery)
    llm = get_model(temperature=0) # Low temp for high accuracy in discovery

    system_instruction = (
        "You are a Senior Legal Counsel. Your first task is to determine if the text is a legal document. "
        "If it is, perform a deep discovery of its substance. "
        "Identify complex 'legalese' terms and provide simple translations. "
        "Expose the hidden risks that a layman might miss. "
        "You MUST return ONLY a JSON object."
    )

    if os.getenv("USE_LOCAL_AI") == "true":
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", "Analyze this text: {contract_text}\n\nFormat: {format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())

        def local_chain(input_data):
            raw_response = (prompt | llm).invoke(input_data)
            content = raw_response.content if isinstance(raw_response, BaseMessage) else str(raw_response)
            sanitized_json = clean_json_text(content) # type: ignore
            try:
                return parser.parse(sanitized_json)
            except Exception as e:
                # Human-like fallback for errors
                return LegalDiscovery(
                    is_legal_document=False,
                    document_type="Unknown",
                    parties=[],
                    complex_terms=[{"term": "Error", "definition": "Could not parse JSON"}],
                    key_obligations=[],
                    hidden_risks=[str(e)]
                )
        return local_chain

    else:
        # Cloud Logic: Uses structured output
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", "Contract Content:\n{contract_text}")
        ])
        return prompt | llm.with_structured_output(LegalDiscovery)
