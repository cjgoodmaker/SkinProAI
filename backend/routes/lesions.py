"""
Lesion Routes - CRUD for lesions and images
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dataclasses import asdict
from pathlib import Path
from PIL import Image
import io

from data.case_store import get_case_store

router = APIRouter()


class CreateLesionRequest(BaseModel):
    name: str
    location: str = ""


class UpdateLesionRequest(BaseModel):
    name: str = None
    location: str = None


# -------------------------------------------------------------------------
# Lesion CRUD
# -------------------------------------------------------------------------

@router.get("/{patient_id}/lesions")
def list_lesions(patient_id: str):
    """List all lesions for a patient"""
    store = get_case_store()

    patient = store.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    lesions = store.list_lesions(patient_id)

    result = []
    for lesion in lesions:
        images = store.list_images(patient_id, lesion.id)
        # Get the most recent image as thumbnail
        latest_image = images[-1] if images else None

        result.append({
            "id": lesion.id,
            "patient_id": lesion.patient_id,
            "name": lesion.name,
            "location": lesion.location,
            "created_at": lesion.created_at,
            "image_count": len(images),
            "latest_image": asdict(latest_image) if latest_image else None
        })

    return {"lesions": result}


@router.post("/{patient_id}/lesions")
def create_lesion(patient_id: str, req: CreateLesionRequest):
    """Create a new lesion for a patient"""
    store = get_case_store()

    patient = store.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    lesion = store.create_lesion(patient_id, req.name, req.location)
    return {
        "lesion": {
            **asdict(lesion),
            "image_count": 0,
            "images": []
        }
    }


@router.get("/{patient_id}/lesions/{lesion_id}")
def get_lesion(patient_id: str, lesion_id: str):
    """Get a lesion with all its images"""
    store = get_case_store()

    lesion = store.get_lesion(patient_id, lesion_id)
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    images = store.list_images(patient_id, lesion_id)

    return {
        "lesion": {
            **asdict(lesion),
            "image_count": len(images),
            "images": [asdict(img) for img in images]
        }
    }


@router.patch("/{patient_id}/lesions/{lesion_id}")
def update_lesion(patient_id: str, lesion_id: str, req: UpdateLesionRequest):
    """Update a lesion's name or location"""
    store = get_case_store()

    lesion = store.get_lesion(patient_id, lesion_id)
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    store.update_lesion(patient_id, lesion_id, req.name, req.location)

    # Return updated lesion
    lesion = store.get_lesion(patient_id, lesion_id)
    images = store.list_images(patient_id, lesion_id)

    return {
        "lesion": {
            **asdict(lesion),
            "image_count": len(images),
            "images": [asdict(img) for img in images]
        }
    }


@router.delete("/{patient_id}/lesions/{lesion_id}")
def delete_lesion(patient_id: str, lesion_id: str):
    """Delete a lesion and all its images"""
    store = get_case_store()

    lesion = store.get_lesion(patient_id, lesion_id)
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    store.delete_lesion(patient_id, lesion_id)
    return {"success": True}


# -------------------------------------------------------------------------
# Image CRUD
# -------------------------------------------------------------------------

@router.post("/{patient_id}/lesions/{lesion_id}/images")
async def upload_image(patient_id: str, lesion_id: str, image: UploadFile = File(...)):
    """Upload a new image to a lesion's timeline"""
    store = get_case_store()

    lesion = store.get_lesion(patient_id, lesion_id)
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    try:
        # Create image record
        img_record = store.add_image(patient_id, lesion_id)

        # Save the actual image file
        pil_image = Image.open(io.BytesIO(await image.read())).convert("RGB")
        image_path = store.save_lesion_image(patient_id, lesion_id, img_record.id, pil_image)

        # Update image record with path
        store.update_image(patient_id, lesion_id, img_record.id, image_path=image_path)

        # Return updated record
        img_record = store.get_image(patient_id, lesion_id, img_record.id)
        return {"image": asdict(img_record)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to upload image: {e}")


@router.get("/{patient_id}/lesions/{lesion_id}/images/{image_id}")
def get_image_record(patient_id: str, lesion_id: str, image_id: str):
    """Get an image record"""
    store = get_case_store()

    img = store.get_image(patient_id, lesion_id, image_id)
    if not img:
        raise HTTPException(status_code=404, detail="Image not found")

    return {"image": asdict(img)}


@router.get("/{patient_id}/lesions/{lesion_id}/images/{image_id}/file")
def get_image_file(patient_id: str, lesion_id: str, image_id: str):
    """Get the actual image file"""
    store = get_case_store()

    img = store.get_image(patient_id, lesion_id, image_id)
    if not img or not img.image_path:
        raise HTTPException(status_code=404, detail="Image not found")

    path = Path(img.image_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(str(path), media_type="image/png")


@router.get("/{patient_id}/lesions/{lesion_id}/images/{image_id}/gradcam")
def get_gradcam_file(patient_id: str, lesion_id: str, image_id: str):
    """Get the GradCAM visualization for an image"""
    store = get_case_store()

    img = store.get_image(patient_id, lesion_id, image_id)
    if not img or not img.gradcam_path:
        raise HTTPException(status_code=404, detail="GradCAM not found")

    path = Path(img.gradcam_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail="GradCAM file not found")

    return FileResponse(str(path), media_type="image/png")


# -------------------------------------------------------------------------
# Chat
# -------------------------------------------------------------------------

@router.get("/{patient_id}/lesions/{lesion_id}/chat")
def get_chat_history(patient_id: str, lesion_id: str):
    """Get chat history for a lesion"""
    store = get_case_store()

    lesion = store.get_lesion(patient_id, lesion_id)
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    messages = store.get_chat_history(patient_id, lesion_id)
    return {"messages": [asdict(m) for m in messages]}


@router.delete("/{patient_id}/lesions/{lesion_id}/chat")
def clear_chat_history(patient_id: str, lesion_id: str):
    """Clear chat history for a lesion"""
    store = get_case_store()

    lesion = store.get_lesion(patient_id, lesion_id)
    if not lesion:
        raise HTTPException(status_code=404, detail="Lesion not found")

    store.clear_chat_history(patient_id, lesion_id)
    return {"success": True}
