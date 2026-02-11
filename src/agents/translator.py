import os
import re
from pydantic import BaseModel, Field
from typing import List
from .get_model import get_model
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.messages import BaseMessage

class SimplifiedSection(BaseModel):
    title: str = Field(description="The legal topic (e.g., 'Your Pay' or 'Leaving the Company')")
    simple_explanation: str = Field(description="Plain English explanation without legalese")
    action_item: str = Field(description="Specific advice for the user based on this clause")


class ExecutiveSummary(BaseModel):
    tldr: str = Field(description="A 2-sentence 'Bottom Line' for the user")
    key_takeaways: list[SimplifiedSection]
    tone_check: str = Field(description="A brief note on whether this contract is 'Employee Friendly' or 'Company Heavy'")


def clean_json_text(text: str) -> str:
    text = re.sub(r"```json\s*|\s*```", "", text)
    match = re.search(r"\{.*\}", text, re.DOTALL)
    return match.group(0) if match else text

def get_translator_agent():
    parser = PydanticOutputParser(pydantic_object=ExecutiveSummary)
    llm = get_model(temperature=0.3) # Slightly higher temp for better writing

    if os.getenv("USE_LOCAL_AI") == "true":
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal coach. You MUST return ONLY JSON. No intro or markdown."),
            ("user", "Simplify this analysis: {analysis_json}\n\nReturn JSON matching: {format_instructions}")
        ]).partial(format_instructions=parser.get_format_instructions())

        def local_chain(input_data):
            raw_response = (prompt | llm).invoke(input_data)
            content = raw_response.content if isinstance(raw_response, BaseMessage) else str(raw_response)
            try:
                sanitized_json = clean_json_text(content)
                return parser.parse(sanitized_json)
            except Exception as e:
                return ExecutiveSummary(
                    tldr="Could not generate summary.",
                    key_takeaways=[],
                    tone_check="Error"
                )
        return local_chain
    else:
        # Cloud logic...
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a legal coach. Simplify the analysis."),
            ("user", "Analysis: {analysis_json}")
        ])
        return prompt | llm.with_structured_output(ExecutiveSummary)



# from pydantic import BaseModel, Field
# from langchain_deepseek import ChatDeepSeek
# from .get_model import get_model
# from langchain_core.prompts import ChatPromptTemplate


# class SimplifiedSection(BaseModel):
#     title: str = Field(description="The legal topic (e.g., 'Your Pay' or 'Leaving the Company')")
#     simple_explanation: str = Field(description="Plain English explanation without legalese")
#     action_item: str = Field(description="Specific advice for the user based on this clause")


# class ExecutiveSummary(BaseModel):
#     tldr: str = Field(description="A 2-sentence 'Bottom Line' for the user")
#     key_takeaways: list[SimplifiedSection]
#     tone_check: str = Field(description="A brief note on whether this contract is 'Employee Friendly' or 'Company Heavy'")


# def get_translator_agent():
#     """
#     Creates a LangChain agent that translates complex legal analysis into
#     clear, actionable advice for non-lawyers.
#     """
#     # Using 'deepseek-reasoner' ensures the 'Why' is processed before the 'How'
#     # llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0.3)
#     llm = get_model(temperature=0.3)
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are a compassionate, expert Legal Career Coach. 
#         Your job is to take complex legal analysis and translate it for a non-lawyer. 
#         Focus on clarity, empowerment, and practical action. 
#         Avoid phrases like 'heretofore' or 'indemnification'—use 'compensation' or 'protection' instead."""),
#         ("user", "Legal Analysis Data: {analysis_json}")
#     ])
    
#     return prompt | llm.with_structured_output(ExecutiveSummary)
