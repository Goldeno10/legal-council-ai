from typing import Annotated, TypedDict, List, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # 'add_messages' ensures new messages are appended to history rather than overwriting it
    messages: Annotated[List[BaseMessage], add_messages]
    raw_text: str
    extracted_data: Optional[dict]
    analysis: Optional[dict]
    final_summary: Optional[dict]
    errors: List[str]
