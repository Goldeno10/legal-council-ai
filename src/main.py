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