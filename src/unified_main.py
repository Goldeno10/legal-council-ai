import os
import asyncio
import json
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict
import threading

from langchain_core.messages import HumanMessage

from src.core.unified_engine import create_legal_engine
from src.utils.parser import parse_legal_document
from src.utils.scrub import anonymize_contract
from src.core.rag_pipeline import LegalRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LegalCouncil AI")
engine = create_legal_engine()
rag_engine = LegalRAG()

# Thread-safe document store so we can keep track
# of uploaded documents and their processed versions across requests
doc_store: Dict[str, str] = {}
doc_store_lock = threading.Lock()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def serialize_data(data):
    """Recursively convert Pydantic models/dicts to JSON-serializable types."""
    if isinstance(data, BaseModel):
        return data.model_dump()
    if isinstance(data, dict):
        return {k: serialize_data(v) for k, v in data.items()}
    if isinstance(data, list):
        return [serialize_data(i) for i in data]
    return data


@app.get("/", response_class=HTMLResponse)
async def get_ui():
    template_path = os.path.join("templates", "unified_index.html")
    with open(template_path, "r") as f:
        return f.read()


@app.post("/analyze")
async def analyze_document(file: UploadFile = File(...)):
    thread_id = str(uuid.uuid4())
    content = await file.read()

    os.makedirs("data/raw", exist_ok=True)
    temp_path = f"data/raw/{thread_id}.pdf"

    with open(temp_path, "wb") as f:
        f.write(content)

    async def event_generator():
        try:
            # Send initial progress
            yield f"data: {json.dumps({'progress': 5, 'message': 'File uploaded'})}\n\n"
            
            # Extract and clean text
            yield f"data: {json.dumps({'progress': 15, 'message': 'Extracting and cleaning text'})}\n\n"
            
            raw_md = parse_legal_document(temp_path)
            safe_md = anonymize_contract(raw_md)
            
            # Thread-safe store update
            with doc_store_lock:
                doc_store[thread_id] = safe_md

            yield f"data: {json.dumps({'progress': 30, 'message': 'Document prepared for analysis'})}\n\n"

            config = {"configurable": {"thread_id": thread_id}}

            initial_state = {
                "messages": [],
                "raw_text": safe_md,
                "is_legal": True,
                "final_summary": None,
                "mode": "analyze",
                "errors": []
            }

            # Run legal validation & risk scan
            yield f"data: {json.dumps({'progress': 40, 'message': 'Running legal validation & risk scan'})}\n\n"

            analysis_complete = False
            brain_data = None
            
            # Stream updates from engine
            async for chunk in engine.astream(initial_state, config=config, stream_mode="updates"): # type: ignore
                node_name = list(chunk.keys())[0]
                data = chunk[node_name]

                if node_name == "brain":
                    if data.get("is_legal") is False:
                        yield f"data: {json.dumps({'progress': 100, 'message': 'Analysis complete'})}\n\n"
                        yield f"data: {json.dumps({'error': 'Not a legal document', 'thread_id': thread_id})}\n\n"
                        return

                    # Store brain data and mark completion
                    brain_data = serialize_data(data)
                    analysis_complete = True

                    # Send the brain data immediately
                    yield f"data: {json.dumps({'node': 'brain', 'update': brain_data, 'progress': 95})}\n\n"

            # Send final completion signals immediately after brain node
            if analysis_complete:
                # Small delay to ensure UI processes the brain data
                await asyncio.sleep(0.1)
                
                # Send final progress and done status together
                final_payload = {
                    'progress': 100, 
                    'message': 'Analysis complete',
                    'status': 'done', 
                    'thread_id': thread_id
                }
                yield f"data: {json.dumps(final_payload)}\n\n"
                
                # Log completion
                logger.info(f"Analysis complete for thread {thread_id}")
            else:
                # Handle case where brain node wasn't found
                yield f"data: {json.dumps({'error': 'Analysis incomplete - no results generated', 'thread_id': thread_id})}\n\n"

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            yield f"data: {json.dumps({'progress': 100, 'message': 'Error occurred', 'error': f'Analysis failed: {str(e)}', 'thread_id': thread_id})}\n\n"
        
        finally:
            # Clean up temp file
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception as e:
                logger.warning(f"Failed to remove temp file {temp_path}: {e}")

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/{thread_id}")
async def chat_stream(thread_id: str, query: str):
    # Check if thread exists
    with doc_store_lock:
        if thread_id not in doc_store:
            raise HTTPException(404, "Session not found")
        doc_text = doc_store[thread_id]  # Get doc for logging/context

    config = {"configurable": {"thread_id": thread_id}}

    input_data = {
        "messages": [HumanMessage(content=query)],
        "mode": "chat",
    }

    async def event_generator():
        try:
            # Send a small initial delay to ensure connection is established
            await asyncio.sleep(0.05)
            
            # Track if we've sent any tokens
            tokens_sent = False
            
            async for event in engine.astream_events(input_data, config, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        tokens_sent = True
                        yield f"data: {json.dumps({'token': content})}\n\n"

                elif kind == "on_tool_start":
                    # Optional: send tool start message
                    tool_name = event.get("name", "unknown")
                    yield f"data: {json.dumps({'tool_start': tool_name})}\n\n"

                elif kind == "on_tool_end":
                    # Optional: yield tool output summary
                    output = event["data"].get("output", {})
                    if output:
                        yield f"data: {json.dumps({'tool_end': 'retrieved context'})}\n\n"

            # Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            
            # Log successful completion
            logger.info(f"Chat completed for thread {thread_id}, tokens sent: {tokens_sent}")

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Chat failed: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.on_event("shutdown")
async def shutdown_event():
    """Clean up resources on shutdown"""
    with doc_store_lock:
        doc_store.clear()
    logger.info("Application shutdown, cleared document store")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)