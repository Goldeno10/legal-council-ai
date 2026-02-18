import os
import re
from pydantic import BaseModel, Field
from typing import List, Dict
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser


class UnifiedLegalResponse(BaseModel):
    is_legal: bool 
    doc_type: str
    
    # We now ask for Markdown-formatted strings
    briefing_md: str = Field(description="Markdown: A warm, empathetic briefing with headers and bold text.")
    glossary_md: str = Field(description="Markdown: List of complex terms defined simply.")
    risks_md: str = Field(description="Markdown: A numbered list of risks with severity and advice.")
    
    verdict: str # Keep this as a single word for the UI tag
    coaches_tip_md: str = Field(description="Markdown: A supportive 'insider' tip.")


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

def get_unified_agent():
    parser = PydanticOutputParser(pydantic_object=UnifiedLegalResponse)
    llm = get_model(temperature=0)

    # THE MASTER PROMPT
    master_instruction = (
        "You are a Senior Legal Counsel and Career Coach. "
        "Analyze the provided text in a single pass to: "
        "1. VALIDATE: Determine if this is a legal document. "
        "2. DISCOVER: Identify complex jargon and define it for a layman. "
        "3. ANALYZE: Apply a risk playbook (Non-competes > 6mo = High Risk, Notice > 3mo = Medium). "
        "4. COACH: Provide a verdict (Sign/Negotiate/Walk) and a supportive tip. "
        "Format each field using Markdown. Use ### for headers, * for bullet points, and ** for emphasis. Do not use JSON keys inside these fields; provide a human-readable narrative."
        "Write in warm, professional, yet approachable tone — like a trusted senior lawyer speaking directly to a founder or early-career professional."
"Use short paragraphs. Prefer active voice. Never write like a robot or use bullet lists inside JSON strings — only use markdown lists when they improve readability."
"""Make section transitions feel natural (e.g. "Now let's look at the main risks..." or "A few important terms you should understand:")."""
    )

    if os.getenv("USE_LOCAL_AI") == "true":
        prompt = ChatPromptTemplate.from_messages([
            ("system", master_instruction),
            ("user", "Document: {contract_text}\n\nFormat: {format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())

        def local_chain(input_text):
            response = (prompt | llm).invoke({"contract_text": input_text})
            # Use the cleaning utility we built earlier
            content = clean_json_text(response.content) 
            return parser.parse(content)
        return local_chain

    else:
        # Cloud Logic: Uses structured output
        prompt = ChatPromptTemplate.from_messages([
            ("system", master_instruction),
            ("user", "Contract Content:\n{contract_text}")
        ])
        return prompt | llm.with_structured_output(UnifiedLegalResponse)
