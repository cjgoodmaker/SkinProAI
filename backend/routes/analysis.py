"""
Analysis Routes - Image analysis with SSE streaming
"""

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from pathlib import Path
import json
import tempfile

from backend.services.analysis_service import get_analysis_service
from data.case_store import get_case_store

router = APIRouter()


@router.get("/gradcam")
def get_gradcam_by_path(path: str = Query(...)):
    """Serve a temp visualization image (GradCAM or comparison overlay)"""
    if not path:
        raise HTTPException(status_code=400, detail="No path provided")

    temp_dir = Path(tempfile.gettempdir()).resolve()
    resolved_path = Path(path).resolve()
    if not str(resolved_path).startswith(str(temp_dir)):
        raise HTTPException(status_code=403, detail="Access denied")

    allowed_suffixes = ("_gradcam.png", "_comparison.png")
    if not any(resolved_path.name.endswith(s) for s in allowed_suffixes):
        raise HTTPException(status_code=400, detail="Invalid image path")

    if resolved_path.exists():
        return FileResponse(str(resolved_path), media_type="image/png")
    raise HTTPException(status_code=404, detail="Image not found")


@router.post("/{patient_id}/lesions/{lesion_id}/images/{image_id}/analyze")
async def analyze_image(
    patient_id: str,
    lesion_id: str,
    image_id: str,
    question: str = Query(None)
):
    """Analyze an image with SSE streaming"""
    store = get_case_store()

    # Verify image exists
    img = store.get_image(patient_id, lesion_id, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")
    if not img.image_path:
        raise HTTPException(status_code=400, detail="Image has no file uploaded")

    service = get_analysis_service()

    async def generate():
        try:
            for chunk in service.analyze(patient_id, lesion_id, image_id, question):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'[ERROR]{str(e)}[/ERROR]')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/{patient_id}/lesions/{lesion_id}/images/{image_id}/confirm")
async def confirm_diagnosis(
    patient_id: str,
    lesion_id: str,
    image_id: str,
    confirmed: bool = Query(...),
    feedback: str = Query(None)
):
    """Confirm or reject diagnosis and get management guidance"""
    service = get_analysis_service()

    async def generate():
        try:
            for chunk in service.confirm(patient_id, lesion_id, image_id, confirmed, feedback):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'[ERROR]{str(e)}[/ERROR]')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/{patient_id}/lesions/{lesion_id}/images/{image_id}/compare")
async def compare_to_previous(
    patient_id: str,
    lesion_id: str,
    image_id: str
):
    """Compare this image to the previous one in the timeline"""
    store = get_case_store()

    # Get current and previous images
    current_img = store.get_image(patient_id, lesion_id, image_id)
    if not current_img:
        raise HTTPException(status_code=404, detail="Image not found")

    previous_img = store.get_previous_image(patient_id, lesion_id, image_id)
    if not previous_img:
        raise HTTPException(status_code=400, detail="No previous image to compare")

    service = get_analysis_service()

    async def generate():
        try:
            for chunk in service.compare_images(
                patient_id, lesion_id,
                previous_img.image_path,
                current_img.image_path,
                image_id
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'[ERROR]{str(e)}[/ERROR]')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )


@router.post("/{patient_id}/lesions/{lesion_id}/chat")
async def chat_message(
    patient_id: str,
    lesion_id: str,
    message: dict
):
    """Send a chat message with SSE streaming response"""
    store = get_case_store()

    lesion = store.get_lesion(patient_id, lesion_id)
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    service = get_analysis_service()
    content = message.get("content", "")

    async def generate():
        try:
            for chunk in service.chat_followup(patient_id, lesion_id, content):
                yield f"data: {json.dumps(chunk)}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: {json.dumps(f'[ERROR]{str(e)}[/ERROR]')}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        }
    )
