"""
Chat Routes - Patient-level chat with image analysis tools
"""

import asyncio
import json
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from data.case_store import get_case_store
from backend.services.chat_service import get_chat_service

router = APIRouter()


@router.get("/{patient_id}/chat")
def get_chat_history(patient_id: str):
    """Get patient-level chat history"""
    store = get_case_store()
    if not store.get_patient(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    messages = store.get_patient_chat_history(patient_id)
    return {"messages": messages}


@router.delete("/{patient_id}/chat")
def clear_chat(patient_id: str):
    """Clear patient-level chat history"""
    store = get_case_store()
    if not store.get_patient(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")
    store.clear_patient_chat_history(patient_id)
    return {"success": True}


@router.post("/{patient_id}/chat")
async def post_chat_message(
    patient_id: str,
    content: str = Form(""),
    image: Optional[UploadFile] = File(None),
):
    """Send a chat message, optionally with an image — SSE streaming response.

    The sync ML generator runs in a background thread so it never blocks the
    event loop.  Events flow through an asyncio.Queue, so each SSE event is
    flushed to the browser the moment it is produced (spinner shows instantly).
    """
    store = get_case_store()
    if not store.get_patient(patient_id):
        raise HTTPException(status_code=404, detail="Patient not found")

    image_bytes = None
    if image and image.filename:
        image_bytes = await image.read()

    chat_service = get_chat_service()

    async def generate():
        loop = asyncio.get_event_loop()
        queue: asyncio.Queue = asyncio.Queue()

        _SENTINEL = object()

        def run_sync():
            try:
                for event in chat_service.stream_chat(patient_id, content, image_bytes):
                    loop.call_soon_threadsafe(queue.put_nowait, event)
            except Exception as e:
                loop.call_soon_threadsafe(
                    queue.put_nowait,
                    {"type": "error", "message": str(e)},
                )
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, _SENTINEL)

        thread = threading.Thread(target=run_sync, daemon=True)
        thread.start()

        while True:
            event = await queue.get()
            if event is _SENTINEL:
                break
            yield f"data: {json.dumps(event)}\n\n"

        yield f"data: {json.dumps({'type': 'done'})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
