import uuid
from typing import (
    Annotated, List, Literal, 
    TypedDict, Optional
)
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import (
    BaseMessage, SystemMessage,
    HumanMessage, AIMessage
)
from langgraph.graph.message import add_messages

from src.agents.unified_extractor import get_unified_agent
from src.agents.get_model import get_model
from src.core.rag_pipeline import LegalRAG


# ----------------------------------------------------------------------
# State Definition
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    raw_text: str
    is_legal: bool
    final_summary: Optional[dict]
    mode: Literal["analyze", "chat"]          # ← controls entry path
    errors: List[str]


# ----------------------------------------------------------------------
# Global Resources (singleton style)
# ----------------------------------------------------------------------
rag_engine = LegalRAG()
rag_tool = rag_engine.as_tool()               # tool is created once


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
def indexer_node(state: AgentState) -> dict:
    """Background indexing of the document (fire-and-forget)."""
    if state.get("raw_text"):
        rag_engine.index_document(state["raw_text"], doc_id=str(uuid.uuid4()))
    return {}


def brain_node(state: AgentState) -> dict:
    """Single-pass legal analysis (validation + summary + risk assessment)."""
    agent = get_unified_agent()
    input_text = state["raw_text"][:15000]

    try:
        result = (
            agent(input_text)
            if callable(agent)
            else agent.invoke({"contract_text": input_text})
        )

        if not getattr(result, "is_legal", False):
            return {
                "is_legal": False,
                "errors": ["Not a recognized legal document."],
            }

        return {
            "is_legal": True,
            "final_summary": result.model_dump(),
            "errors": [],
        }

    except Exception as e:
        return {"errors": [f"Brain error: {str(e)}"]}

def chat_agent(state: AgentState) -> dict:
    """Conversational Legal Coach with optional contract retrieval tool."""
    llm = get_model(temperature=0.75, format=None)
    llm_with_tools = llm.bind_tools([rag_tool])

    summary = state.get("final_summary", {})
    doc_type = summary.get("doc_type", "the agreement")
    verdict = summary.get("verdict", "N/A")

    system_content = f"""You are a supportive Legal Career Coach.
        Background (reference only, do NOT repeat):
        - Document: {doc_type}
        - Recommendation: {verdict}

        You have access to a tool to search the contract text.
        If the user asks about specific clauses, obligations, definitions, 
        or risks in the document, use the tool to find relevant excerpts and quote them in your answer.
        Do NOT use the tool for general legal advice or negotiation strategy unless it directly references the document.
        Do NOT hallucinate contract text — if the tool doesn't return relevant excerpts,
        answer based on your general legal knowledge without making up quotes.
        Do NOT tell the user to/about use the tool — use it internally as needed to find information.
        Use it ONLY when needed for specific clauses or wording.
        NEVER output raw XML, <function_calls>, <invoke>, or any tags — the system handles tool calls automatically.
        Respond in plain, natural English only. If calling a tool, do so internally without formatting."""

    messages = [SystemMessage(content=system_content)] + state["messages"]

    # Retry loop for malformed outputs (max 2 tries)
    for _ in range(2):
        response = llm_with_tools.invoke(messages)
        
        # Check for hallucinated XML
        if isinstance(response.content, str) and ("<function_calls>" in response.content or "<invoke>" in response.content):
            # Fallback: add correction message and retry
            messages.append(response)
            messages.append(SystemMessage("Your last response had invalid XML. Respond in plain text only, no tags. Use tools internally if needed."))
            continue
        
        # Valid — return
        return {"messages": [response]}
    
    # Fallback after retries
    fallback_response = AIMessage(content="Sorry, I'm having trouble thinking clearly. Let's try that question again?")
    return {"messages": [fallback_response]}


# ----------------------------------------------------------------------
# Router
# ----------------------------------------------------------------------
def route_entry(state: AgentState) -> Literal["brain", "chat_agent"]:
    """Decide which path to take based on mode + guard conditions."""
    mode = state.get("mode")

    if mode == "analyze":
        return "brain"

    if mode == "chat":
        # Guard: chat is only allowed after successful analysis
        if state.get("final_summary") and not state.get("errors"):
            return "chat_agent"
        # Fallback: if chat is requested but analysis failed → run analysis first
        # (prevents broken chats on bad documents)
        return "brain"

    # Default fallback (should never reach here with proper input)
    return "brain"


# ----------------------------------------------------------------------
# Graph Construction
# ----------------------------------------------------------------------
def create_legal_engine():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("brain", brain_node)
    workflow.add_node("indexer", indexer_node)
    workflow.add_node("chat_agent", chat_agent)
    workflow.add_node("tools", ToolNode(tools=[rag_tool]))

    # Entry router
    workflow.add_node("router", lambda state: {"mode": state.get("mode")})
    workflow.set_entry_point("router")

    # Router decides next node
    workflow.add_conditional_edges(
        "router",
        route_entry,
        {"brain": "brain", "chat_agent": "chat_agent"},
    )

    # Analysis path (one-time)
    workflow.add_edge("brain", "indexer")
    workflow.add_edge("indexer", END)

    # Chat path → ReAct loop
    workflow.add_conditional_edges(
        "chat_agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "chat_agent")

    return workflow.compile(checkpointer=InMemorySaver())
