import uuid
from typing import Annotated, List, TypedDict, Optional, Literal
from langchain_core.messages import SystemMessage

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

# def extractor_node(state: AgentState):
#     # Guard against empty input before calling LLM
#     if not state.get("raw_text"):
#         return {"errors": ["No text provided for extraction"]}
        
#     agent = get_extraction_agent()
#     try:
#         result = agent.invoke({"contract_text": state["raw_text"][:12000]})
#         return {"extracted_data": result}
#     except Exception as e:
#         return {"errors": [f"Extraction Error: {str(e)}"]}
def extractor_node(state: AgentState):
    agent = get_extraction_agent()
    try:
        input_data = {"contract_text": state["raw_text"][:30000]}
        
        # Check if the agent is a function (Local mode) or Runnable (Cloud mode)
        if callable(agent) and not hasattr(agent, "invoke"):
            # It's our custom local_chain function
            result = agent(input_data)
        else:
            # It's a standard LangChain Runnable
            result = agent.invoke(input_data)
            
        # Convert Pydantic to dict for the state
        return {"extracted_data": result.model_dump() if hasattr(result, "model_dump") else result.dict() if hasattr(result, "dict") else result}
        
    except Exception as e:
        print(f"Node Error: {e}")
        return {"errors": [f"Extraction Error: {str(e)}"]}


def analyzer_node(state: AgentState):
    agent = get_analyzer_agent()
    result = agent.invoke({"extracted_json": state["extracted_data"]})
    return {"analysis": result}

# def translator_node(state: AgentState):
#     agent = get_translator_agent()
#     result = agent.invoke({"analysis_json": state["analysis"]})
#     return {"final_summary": result}

def translator_node(state: AgentState):
    if state.get("errors"): return state
    
    agent = get_translator_agent()
    input_data = {"analysis_json": state["analysis"]}
    
    try:
        # HANDLE BOTH FUNCTION AND RUNNABLE
        if callable(agent) and not hasattr(agent, "invoke"):
            result = agent(input_data)
        else:
            result = agent.invoke(input_data) # type: ignore
        
        # Safely extract dict-like data from various return types
        model_dump_fn = getattr(result, "model_dump", None)
        dict_fn = getattr(result, "dict", None)
        if callable(model_dump_fn):
            final = model_dump_fn()
        elif callable(dict_fn):
            final = dict_fn()
        elif isinstance(result, dict):
            final = result
        else:
            final = result
            
        return {"final_summary": final}
    except Exception as e:
        print(f"Translator Error: {e}")
        return {"errors": [f"Translator Error: {str(e)}"]}


def chat_node(state: AgentState):
    user_query = state["messages"][-1].content
    
    # 1. RAG Retrieval
    relevant_chunks = rag_engine.query_contract(user_query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])
    
    # 2. Contextual Summary (Prevent JSON Mimicry)
    # We pull the high-level findings from the state but present them as text
    analysis = state.get("analysis", {})
    pros = ", ".join(analysis.get("pros", []))
    cons = ", ".join(analysis.get("cons", []))
    
    # 3. Enhanced Professional Prompt
    system_prompt = f"""
    You are a professional Legal Career Coach. 
    Use the following contract snippets and our risk analysis to answer the user's question.

    RISK ANALYSIS SUMMARY:
    - Pros found: {pros if pros else "None identified"}
    - Risks found: {cons if cons else "None identified"}

    CONTRACT SNIPPETS:
    {context_text}

    INSTRUCTIONS:
    1. Respond in PLAIN ENGLISH. 
    2. NEVER output JSON, code blocks, or raw data structures.
    3. Use bullet points for clarity.
    4. If the answer is not in the snippets, say you don't have enough information from the document.
    """

    # 4. Use the model factory (ensure temperature is higher for natural speech)
    llm = get_model(temperature=0.7) 
    
    # 5. Build the message chain
    messages = [SystemMessage(content=system_prompt)] + state["messages"]
    
    response = llm.invoke(messages)
    return {"messages": [response]}


# def chat_node(state: AgentState):
#     user_query = state["messages"][-1].content
#     relevant_chunks = rag_engine.query_contract(user_query)
#     context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])
    
#     llm = get_model(model="deepseek-chat")
#     system_msg = f"Answer using context:\n{context_text}"
    
#     response = llm.invoke([{"role": "system", "content": system_msg}] + state["messages"])
#     return {"messages": [response]}


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