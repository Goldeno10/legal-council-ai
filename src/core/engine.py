import operator
from typing import Annotated, List, Union, TypedDict, Optional
import uuid

from langgraph.graph import StateGraph, END
from langchain_deepseek import ChatDeepSeek
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.messages import BaseMessage, HumanMessage

from src.agents.extractor import get_extraction_agent
from src.agents.analyzer import get_analyzer_agent
from src.agents.translator import get_translator_agent
from src.agents.get_model import get_model

from src.core.rag_pipeline import LegalRAG



# Initialize the RAG engine globally or within the node
rag_engine = LegalRAG()


# 1. Define the State Object
class AgentState(TypedDict):
    # 'add_messages' allows the state to append new chat history
    messages: Annotated[List[BaseMessage], operator.add]
    raw_text: str
    extracted_data: Optional[dict]
    analysis: Optional[dict]
    final_summary: Optional[dict]
    errors: List[str]



def indexing_node(state: AgentState):
    """
    Takes the parsed markdown and stores it in the vector database
    for semantic retrieval during chat.
    """
    if state.get("errors"): return state
    
    # We use a unique ID (like a thread_id) to keep documents separate
    doc_id = str(uuid.uuid4()) 
    rag_engine.index_document(state["raw_text"], doc_id=doc_id)
    
    return {"errors": []} # Success



# 2. Define the Node Functions
def extractor_node(state: AgentState):
    agent = get_extraction_agent()
    try:
        # Use only the first 100k chars for extraction safety
        result = agent.invoke({"contract_text": state["raw_text"][:100000]})
        return {"extracted_data": result.dict()}
    except Exception as e:
        return {"errors": [f"Extraction Error: {str(e)}"]}

def analyzer_node(state: AgentState):
    if state.get("errors"): return state
    agent = get_analyzer_agent()
    result = agent.invoke({"extracted_json": state["extracted_data"]})
    return {"analysis": result.dict()}

def translator_node(state: AgentState):
    if state.get("errors"): return state
    agent = get_translator_agent()
    result = agent.invoke({"analysis_json": state["analysis"]})
    return {"final_summary": result.dict()}

# def chat_node(state: AgentState):
#     """
#     Handles follow-up questions using the existing context.
#     """
#     llm = ChatDeepSeek(model="deepseek-chat")
    
#     # Contextualize the chat with the already analyzed data
#     context = f"Contract Data: {state['extracted_data']}\nRisk Analysis: {state['analysis']}"
#     system_msg = f"You are a legal assistant. Answer questions based on this context: {context}"
    
#     messages = [{"role": "system", "content": system_msg}] + state["messages"]
#     response = llm.invoke(messages)
#     return {"messages": [response]}

def chat_node(state: AgentState):
    """
    A RAG-powered chat node.
    """
    user_query = state["messages"][-1].content
    
    # 1. RETRIEVAL: Find only the relevant clauses
    relevant_chunks = rag_engine.query_contract(user_query)
    context_text = "\n\n".join([doc.page_content for doc in relevant_chunks])
    
    # 2. GENERATION: Ground the answer in the retrieved context
    # llm = ChatDeepSeek(model="deepseek-chat")
    llm = get_model(model="deepseek-chat")

    system_msg = f"""
    You are a legal assistant. Answer the user's question ONLY using the provided context.
    If the answer isn't in the context, say you don't know.
    
    CONTEXT:
    {context_text}
    """
    
    response = llm.invoke([{"role": "system", "content": system_msg}] + state["messages"])
    return {"messages": [response]}


# 3. Construct the Graph
def create_legal_engine():
    workflow = StateGraph(AgentState)

    # Add Nodes
    workflow.add_node("extractor", extractor_node)
    workflow.add_node("analyzer", analyzer_node)
    workflow.add_node("indexer", indexing_node) # NEW
    workflow.add_node("translator", translator_node)
    workflow.add_node("chat", chat_node)

    # Define Flow
    workflow.set_entry_point("extractor")
    workflow.add_edge("extractor", "analyzer")
    workflow.add_edge("analyzer", "indexer")    # NEW
    workflow.add_edge("indexer", "translator")  # Indexing happens in parallel/sequence
    workflow.add_edge("translator", END)

    # Chat logic is usually triggered by a specific entry point in production
    workflow.add_edge("chat", END)

    # Add persistence (Checkpointer)
    memory = InMemorySaver()
    return workflow.compile(checkpointer=memory)
