from pydantic import BaseModel, Field
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate


class SimplifiedSection(BaseModel):
    title: str = Field(description="The legal topic (e.g., 'Your Pay' or 'Leaving the Company')")
    simple_explanation: str = Field(description="Plain English explanation without legalese")
    action_item: str = Field(description="Specific advice for the user based on this clause")


class ExecutiveSummary(BaseModel):
    tldr: str = Field(description="A 2-sentence 'Bottom Line' for the user")
    key_takeaways: list[SimplifiedSection]
    tone_check: str = Field(description="A brief note on whether this contract is 'Employee Friendly' or 'Company Heavy'")


def get_translator_agent():
    """
    Creates a LangChain agent that translates complex legal analysis into
    clear, actionable advice for non-lawyers.
    """
    # Using 'deepseek-reasoner' ensures the 'Why' is processed before the 'How'
    llm = ChatDeepSeek(model="deepseek-reasoner", temperature=0.3)
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a compassionate, expert Legal Career Coach. 
        Your job is to take complex legal analysis and translate it for a non-lawyer. 
        Focus on clarity, empowerment, and practical action. 
        Avoid phrases like 'heretofore' or 'indemnification'—use 'compensation' or 'protection' instead."""),
        ("user", "Legal Analysis Data: {analysis_json}")
    ])
    
    return prompt | llm.with_structured_output(ExecutiveSummary)
