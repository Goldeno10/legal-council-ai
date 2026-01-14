import json
import uuid
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse
from langchain_core.messages import HumanMessage
from src.core.engine import create_legal_engine
from src.utils.parser import parse_legal_document
from src.utils.scrub import anonymize_contract


app = FastAPI(title="LegalCouncil AI")
engine = create_legal_engine()
doc_store = {}

@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    """
    Analyzes the doc and streams step-by-step progress to the client.
    """
    thread_id = str(uuid.uuid4())
    content = await file.read()
    temp_path = f"data/raw/{thread_id}.pdf"
    
    with open(temp_path, "wb") as f:
        f.write(content)

    # In-memory context storage
    raw_md = parse_legal_document(temp_path)
    safe_md = anonymize_contract(raw_md)
    doc_store[thread_id] = safe_md

    async def event_generator():
        config = {"configurable": {"thread_id": thread_id}}
        initial_state = {"raw_text": safe_md, "messages": [], "errors": []}

        # Use .astream with stream_mode="updates" to capture node completion
        # stream_mode="messages" can be used for raw LLM token streaming
        async for chunk in engine.astream(initial_state, config=config, stream_mode="updates"):
            # 'chunk' is a dict mapping node names to their state updates
            node_name = list(chunk.keys())[0]
            data = chunk[node_name]
            
            # Send an SSE event for each completed node
            yield f"data: {json.dumps({'node': node_name, 'status': 'completed', 'update': data})}\n\n"

        yield f"data: {json.dumps({'status': 'done', 'thread_id': thread_id})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.post("/chat/{thread_id}")
async def chat_stream(thread_id: str, query: str):
    """
    Streams a chat response token-by-token.
    """
    if thread_id not in doc_store:
        raise HTTPException(status_code=404, detail="Session not found")

    async def chat_generator():
        config = {"configurable": {"thread_id": thread_id}}
        input_data = {"messages": [HumanMessage(content=query)]}

        # Stream messages specifically (for token-by-token chat)
        async for msg, metadata in engine.astream(input_data, config=config, stream_mode="messages"):
            if msg.content:
                yield f"data: {json.dumps({'token': msg.content})}\n\n"

    return StreamingResponse(chat_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



# import uuid
# import json
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse

# from src.utils.parser import parse_legal_document
# from src.utils.scrub import anonymize_contract
# from src.core.engine import create_legal_engine
# from langchain_core.messages import HumanMessage

# app = FastAPI(title="LegalCouncil AI")
# engine = create_legal_engine()

# # Dictionary to map thread_ids to document content (in-memory for MVP)
# doc_store = {}

# @app.post("/upload")
# async def upload_and_analyze(file: UploadFile = File(...)):
#     """
#     Initial processing: Parse -> Scrub -> Agentic Pipeline
#     """
#     thread_id = str(uuid.uuid4())
    
#     # 1. Parse & Privacy Scrubbing
#     content = await file.read()
#     temp_path = f"data/raw/{thread_id}.pdf"
#     with open(temp_path, "wb") as f:
#         f.write(content)
        
#     raw_md = parse_legal_document(temp_path)
#     safe_md = anonymize_contract(raw_md)
    
#     # Store text for the chat context
#     doc_store[thread_id] = safe_md

#     # 2. Run the Graph
#     config = {"configurable": {"thread_id": thread_id}}
#     initial_state = {
#         "raw_text": safe_md,
#         "messages": [],
#         "errors": []
#     }
    
#     # We use 'invoke' for the initial full-scale analysis
#     result = await engine.ainvoke(initial_state, config=config)
    
#     return {
#         "thread_id": thread_id,
#         "analysis": result["analysis"],
#         "summary": result["final_summary"]
#     }

# @app.post("/chat/{thread_id}")
# async def chat(thread_id: str, query: str):
#     """
#     Follow-up chat using the persistent thread_id memory.
#     """
#     if thread_id not in doc_store:
#         raise HTTPException(status_code=404, detail="Session expired or not found")

#     config = {"configurable": {"thread_id": thread_id}}
    
#     # We only send the new message; LangGraph retrieves the previous state
#     input_data = {"messages": [HumanMessage(content=query)]}
    
#     # Execute the 'chat' node specifically
#     result = await engine.ainvoke(input_data, config=config, interrupt_before=["extractor"])
    
#     return {"response": result["messages"][-1].content}

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)






# import uuid
# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.responses import StreamingResponse
# from src.utils.parser import parse_legal_document
# from src.utils.scrub import anonymize_contract
# from src.core.engine import create_legal_engine
# import json

# app = FastAPI(title="LegalCouncil AI API", version="1.0.0")
# engine = create_legal_engine()

# @app.post("/analyze")
# async def analyze_document(file: UploadFile = File(...)):
#     """
#     High-performance endpoint that streams the agentic workflow 
#     progress to the frontend.
#     """
#     # 1. Validation & Save
#     if not file.filename.endswith('.pdf'):
#         raise HTTPException(status_code=400, detail="Only PDF documents are supported.")
    
#     temp_path = f"data/raw/{uuid.uuid4()}.pdf"
#     with open(temp_path, "wb") as buffer:
#         buffer.write(await file.read())

#     # 2. Ingestion & Anonymization (Google-Standard Privacy)
#     raw_md = parse_legal_document(temp_path)
#     safe_md = anonymize_contract(raw_md)

#     # 3. Stream the Multi-Agent Graph
#     async def event_generator():
#         initial_state = {
#             "raw_text": safe_md,
#             "extracted_data": None,
#             "analysis": None,
#             "final_summary": None,
#             "errors": []
#         }

#         # We stream 'updates' to show the UI which agent is currently working
#         async for event in engine.astream(initial_state):
#             # 'event' contains the output of the node that just finished
#             node_name = list(event.keys())[0]
#             data = event[node_name]
            
#             yield f"data: {json.dumps({'step': node_name, 'status': 'completed'})}\n\n"
            
#             # If we reached the end, send the final payload
#             if "final_summary" in data and data["final_summary"]:
#                 yield f"data: {json.dumps({'result': data['final_summary']})}\n\n"

#     return StreamingResponse(event_generator(), media_type="text/event-stream")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)
