import os
import asyncio
import json
import uuid
import logging
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain_core.messages import HumanMessage

from src.core.engine import create_legal_engine
from src.utils.parser import parse_legal_document
from src.utils.scrub import anonymize_contract
from src.core.rag_pipeline import LegalRAG

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LegalCouncil AI")
engine = create_legal_engine()
rag_engine = LegalRAG()
doc_store = {}  # TODO: Consider making thread-safe with locks for production

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
        yield f"data: {json.dumps({'progress': 5, 'message': 'File uploaded'})}\n\n"
        await asyncio.sleep(0.2)

        yield f"data: {json.dumps({'progress': 15, 'message': 'Extracting and cleaning text'})}\n\n"
        
        raw_md = parse_legal_document(temp_path)
        safe_md = anonymize_contract(raw_md)
        doc_store[thread_id] = safe_md

        yield f"data: {json.dumps({'progress': 30, 'message': 'Document prepared for analysis'})}\n\n"
        await asyncio.sleep(0.3)

        config = {"configurable": {"thread_id": thread_id}}

        initial_state = {
            "messages": [],
            "raw_text": safe_md,
            "is_legal": True,
            "final_summary": None,
            "mode": "analyze",
            "errors": []
        }

        try:
            yield f"data: {json.dumps({'progress': 40, 'message': 'Running legal validation & risk scan'})}\n\n"

            async for chunk in engine.astream(initial_state, config=config, stream_mode="updates"):
                node_name = list(chunk.keys())[0]
                data = chunk[node_name]

                if node_name == "validator" and data.get("is_legal") is False:
                    error_msg = data.get("errors", ["Not a legal document."])[0]
                    yield f"data: {json.dumps({'progress': 90, 'message': 'Analysis complete'})}\n\n"
                    yield f"data: {json.dumps({'error': error_msg})}\n\n"
                    return

                yield f"data: {json.dumps({'progress': 85, 'message': f'Processing {node_name} node'})}\n\n"
                await asyncio.sleep(0.4)

                yield f"data: {json.dumps({'node': node_name, 'update': serialize_data(data)})}\n\n"

            yield f"data: {json.dumps({'progress': 100, 'message': 'Analysis complete'})}\n\n"
            yield f"data: {json.dumps({'status': 'done', 'thread_id': thread_id})}\n\n"

        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            yield f"data: {json.dumps({'progress': 100, 'message': 'Error occurred'})}\n\n"
            yield f"data: {json.dumps({'error': f'Analysis failed: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/chat/{thread_id}")
async def chat_stream(thread_id: str, query: str):
    if thread_id not in doc_store:
        raise HTTPException(404, "Session not found")

    config = {"configurable": {"thread_id": thread_id}}

    input_data = {
        "messages": [HumanMessage(content=query)],
        "mode": "chat",
    }

    async def event_generator():
        try:
            async for event in engine.astream_events(input_data, config, version="v2"):
                kind = event["event"]

                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        yield f"data: {json.dumps({'token': content})}\n\n"

                elif kind == "on_tool_start":
                    yield f"data: {json.dumps({'message': 'Searching contract...'})}\n\n"

                elif kind == "on_tool_end":
                    output = event["data"]["output"]
                    yield f"data: {json.dumps({'message': 'Retrieved context', 'details': output})}\n\n"

            yield f"data: {json.dumps({'done': True})}\n\n"

        except Exception as e:
            logger.error(f"Chat error: {str(e)}")
            yield f"data: {json.dumps({'error': f'Chat failed: {str(e)}'})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)