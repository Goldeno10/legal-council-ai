import os
import re
from pydantic import BaseModel, Field
from typing import List
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage


class SimplifiedSection(BaseModel):
    title: str = Field(description="The human-friendly topic name")
    simple_explanation: str = Field(description="What this means for your daily life, in plain English")
    action_item: str = Field(description="What you should actually do or say next")


class ExecutiveSummary(BaseModel):
    tldr: str = Field(description="A warm, empathetic 2-sentence summary of the situation")
    key_takeaways: List[SimplifiedSection]
    coaches_tip: str = Field(description="An 'insider' tip on how to handle this specific deal human-to-human")
    tone_check: str = Field(description="A descriptive mood (e.g., 'Aggressively One-sided' or 'Fair & Collaborative')")


def clean_json_text(text: str) -> str:
    text = re.sub(r"```json\s*|\s*```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

def get_translator_agent():
    parser = PydanticOutputParser(pydantic_object=ExecutiveSummary)
    # 0.5 temperature allows for more natural, varied human language
    llm = get_model(temperature=0.5)

    system_instruction = (
        "You are a warm, expert Legal Career Coach. "
        "Your goal is to make the user feel empowered, not overwhelmed. "
        "Translate the complex discovery and legal analysis into a supportive brief. "
        "Use analogies where helpful and avoid all legal jargon. "
        "If you see a risk, explain it as a 'protection' the user deserves. "
        "You MUST return ONLY a JSON object."
    )

    if os.getenv("USE_LOCAL_AI") == "true":
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", "Analysis to Simplify: {analysis_json}\n\nFormat as JSON: {format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())

        def local_chain(input_data):
            raw_response = (prompt | llm).invoke(input_data)
            content = raw_response.content if isinstance(raw_response, BaseMessage) else str(raw_response)
            try:
                sanitized_json = clean_json_text(content) # type: ignore
                return parser.parse(sanitized_json)
            except Exception as e:
                return ExecutiveSummary(
                    tldr="I'm having a little trouble summarizing this, but let's look at the details together.",
                    key_takeaways=[],
                    coaches_tip="Take a breath and read the sections below carefully.",
                    tone_check="Undetermined"
                )
        return local_chain
    else:
        # Cloud logic (GPT/DeepSeek)
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_instruction),
            ("user", "Analysis to Simplify: {analysis_json}")
        ])
        return prompt | llm.with_structured_output(ExecutiveSummary)