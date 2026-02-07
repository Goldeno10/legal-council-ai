from typing import Annotated, List, TypedDict, Optional, Literal
import uuid

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.message import add_messages

from src.agents.extractor import get_extraction_agent
from src.agents.analyzer import get_analyzer_agent
from src.agents.translator import get_translator_agent
from src.agents.get_model import get_model
from src.core.rag_pipeline import LegalRAG

# Initialize the RAG engine
rag_engine = LegalRAG()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    raw_text: str
    extracted_data: Optional[dict]
    analysis: Optional[dict]
    final_summary: Optional[dict]
    errors: List[str]

# --- ROUTER LOGIC ---
def should_continue(state: AgentState) -> Literal["continue", "end"]:
    """
    Determines if the graph should proceed or stop based on errors.
    """
    if state.get("errors") and len(state["errors"]) > 0:
        return "end"
    return "continue"

# --- NODES ---
def indexing_node(state: AgentState):
    raw_text = state.get("raw_text", "").strip()
    if not raw_text:
        return {"errors": ["No text provided for indexing"]}
    
    doc_id = str(uuid.uuid4()) 
    # Use the [Chroma Documentation](https://docs.trychroma.com) approach for safety
    rag_engine.index_document(raw_text, doc_id=doc_id)
    return {"errors": []}

def extractor_node(state: AgentState):
    # Guard against empty input before calling LLM
    if not state.get("raw_text"):
        return {"errors": ["No text provided for extraction"]}
        
    agent = get_extraction_agent()
    try:
        result = agent.invoke({"contract_text": state["raw_text"][:100000]})
        return {"extracted_data": result}
    except Exception as e:
        return {"errors": [f"Extraction Error: {str(e)}"]}

def analyzer_node(state: AgentState):
    agent = get_analyzer_agent()
    result = agent.invoke({"extracted_json": state["extracted_data"]})
    return {"analysis": result}

def translator_node(state: AgentState):
    agent = get_translator_agent()
    result = agent.invoke({"analysis_json": state["analysis"]})
    return {"final_summary": result}

def chat_node(state: AgentState):
    user_query = state["messages"][-1].content
    relevant_chunks = rag_engine.query_contract(user_query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])
    
    llm = get_model(model="deepseek-chat")
    system_msg = f"Answer using context:\n{context_text}"
    
    response = llm.invoke([{"role": "system", "content": system_msg}] + state["messages"])
    return {"messages": [response]}

# --- CONSTRUCT THE GRAPH ---
def create_legal_engine():
    workflow = StateGraph(AgentState)

    workflow.add_node("indexer", indexing_node)
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("translator", translator_node)
    workflow.add_node("chat", chat_node)

    # UPDATED FLOW: Use conditional edges to check for errors after every step
    workflow.set_entry_point("indexer")
    
    workflow.add_conditional_edges(
        "indexer",
        should_continue,
        {"continue": "extractor", "end": END}
    )
    
    workflow.add_conditional_edges(
        "extractor",
        should_continue,
        {"continue": "analyzer", "end": END}
    )
    
    workflow.add_conditional_edges(
        "analyzer",
        should_continue,
        {"continue": "translator", "end": END}
    )
    
    workflow.add_edge("translator", END)
    workflow.add_edge("chat", END)

    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)

if __name__ == "__main__":
    engine = create_legal_engine()
    # This generates a text string in Mermaid format
    print("\n--- COPY THE CODE BELOW ---")
    print(engine.get_graph().draw_mermaid())
    print("--- END OF CODE ---\n")