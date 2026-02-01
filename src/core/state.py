from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# 1. Define the State Object
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    raw_text: str
    extracted_data: Optional[dict]
    analysis: Optional[dict]
    final_summary: Optional[dict]
    errors: List[str]
