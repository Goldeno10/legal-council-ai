import uuid
from typing import Annotated, List, Literal, TypedDict, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import BaseMessage, SystemMessage
from langgraph.graph.message import add_messages

from src.agents.extractor import get_discovery_agent
from src.agents.analyzer import get_analyzer_agent
from src.agents.translator import get_translator_agent
from src.agents.get_model import get_model
from src.core.rag_pipeline import LegalRAG


# ----------------------------------------------------------------------
# State Definition
# ----------------------------------------------------------------------
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    raw_text: str
    discovery: Optional[dict]
    analysis: Optional[dict]
    final_summary: Optional[dict]
    mode: Literal["analyze", "chat"]  # Controls entry path
    is_legal: bool
    errors: List[str]


# ----------------------------------------------------------------------
# Global Resources
# ----------------------------------------------------------------------
rag_engine = LegalRAG()
rag_tool = rag_engine.as_tool()  # Tool instantiated once


# ----------------------------------------------------------------------
# Nodes
# ----------------------------------------------------------------------
def validator_node(state: AgentState) -> dict:
    """Initial validation: Check if the document is legal."""
    llm = get_model(temperature=0)
    prompt = f"""You are a legal gatekeeper. Analyze the following text snippet.
    Is this a legal document (contract, NDA, lease, etc.)? 
    Respond with exactly one word: 'YES' or 'NO'.
    
    TEXT: {state['raw_text'][:2000]}"""
    
    response = llm.invoke(prompt)
    is_legal = "YES" in response.content.upper()
    
    return {
        "is_legal": is_legal, 
        "errors": [] if is_legal else ["The uploaded file does not appear to be a legal document."]
    }


def indexer_node(state: AgentState) -> dict:
    """Background indexing for RAG (fire-and-forget)."""
    if state.get("raw_text"):
        rag_engine.index_document(state["raw_text"], doc_id=str(uuid.uuid4()))
    return {}


def discovery_node(state: AgentState) -> dict:
    """Extract key elements and jargon from the document."""
    agent = get_discovery_agent()
    input_data = {"contract_text": state["raw_text"][:30000]}
    
    try:
        result = agent(input_data) if callable(agent) else agent.invoke(input_data)
        return {"discovery": result.model_dump() if hasattr(result, "model_dump") else result}
    except Exception as e:
        return {"errors": [f"Discovery error: {str(e)}"]}


def analyzer_node(state: AgentState) -> dict:
    """Assess risks and provide strategic analysis."""
    agent = get_analyzer_agent()
    input_data = {"extracted_json": state["discovery"]}
    
    try:
        result = agent(input_data) if callable(agent) else agent.invoke(input_data)
        return {"analysis": result.model_dump() if hasattr(result, "model_dump") else result}
    except Exception as e:
        return {"errors": [f"Analysis error: {str(e)}"]}


def translator_node(state: AgentState) -> dict:
    """Synthesize discovery and analysis into a human-friendly summary."""
    agent = get_translator_agent()
    input_data = {
        "analysis_json": {
            "discovery": state["discovery"],
            "risks": state["analysis"]
        }
    }
    
    try:
        result = agent(input_data) if callable(agent) else agent.invoke(input_data)
        return {"final_summary": result.model_dump() if hasattr(result, "model_dump") else result}
    except Exception as e:
        return {"errors": [f"Translation error: {str(e)}"]}


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

You have access to a tool that searches the actual contract text.
Use it when the question is about specific clauses, definitions, obligations, or wording in THIS document.
Do NOT use it for general legal knowledge or negotiation tactics unless they directly relate to the contract.

Answer naturally, warmly, in plain English. Be encouraging and actionable.
NEVER output XML, tags, or raw function calls — the system handles tool calls automatically."""

    messages = [SystemMessage(content=system_content)] + state["messages"]

    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


# ----------------------------------------------------------------------
# Routers
# ----------------------------------------------------------------------
def route_after_validation(state: AgentState) -> Literal["discovery", "end"]:
    """Continue analysis only if validated as legal."""
    if state.get("is_legal") and not state.get("errors"):
        return "discovery"
    return "end"


def route_entry(state: AgentState) -> Literal["validator", "chat_agent"]:
    """Decide starting point based on mode, with guard for chat."""
    mode = state.get("mode", "analyze")

    if mode == "analyze":
        return "validator"

    if mode == "chat":
        # Guard: chat requires successful prior analysis
        if state.get("final_summary") and not state.get("errors"):
            return "chat_agent"
        # Fallback: run analysis if chat requested but state invalid
        return "validator"

    # Default to analysis
    return "validator"


# ----------------------------------------------------------------------
# Graph Construction
# ----------------------------------------------------------------------
def create_legal_engine():
    workflow = StateGraph(AgentState)

    # Nodes
    workflow.add_node("validator", validator_node)
    workflow.add_node("indexer", indexer_node)
    workflow.add_node("discovery", discovery_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("chat_agent", chat_agent)
    workflow.add_node("tools", ToolNode(tools=[rag_tool]))

    # Entry router
    workflow.add_node("router", lambda state: {"mode": state.get("mode")})
    workflow.set_entry_point("router")

    # Route to appropriate start
    workflow.add_conditional_edges(
        "router",
        route_entry,
        {"validator": "validator", "chat_agent": "chat_agent"},
    )

    # Analysis path (multi-node)
    workflow.add_conditional_edges(
        "validator",
        route_after_validation,
        {"discovery": "discovery", "end": END},
    )
    workflow.add_edge("validator", "indexer")  # Index in parallel regardless
    workflow.add_edge("discovery", "analyzer")
    workflow.add_edge("analyzer", "translator")
    workflow.add_edge("translator", END)
    workflow.add_edge("indexer", END)  # Indexer ends independently

    # Chat path → ReAct loop
    workflow.add_conditional_edges(
        "chat_agent",
        tools_condition,
        {"tools": "tools", END: END},
    )
    workflow.add_edge("tools", "chat_agent")

    return workflow.compile(checkpointer=InMemorySaver())