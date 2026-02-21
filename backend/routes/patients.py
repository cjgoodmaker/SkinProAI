"""
Patient Routes - CRUD for patients
"""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dataclasses import asdict

from data.case_store import get_case_store

router = APIRouter()


class CreatePatientRequest(BaseModel):
    name: str


@router.get("")
def list_patients():
    """List all patients with lesion counts"""
    store = get_case_store()
    patients = store.list_patients()

    result = []
    for p in patients:
        result.append({
            **asdict(p),
            "lesion_count": store.get_patient_lesion_count(p.id)
        })

    return {"patients": result}


@router.post("")
def create_patient(req: CreatePatientRequest):
    """Create a new patient"""
    store = get_case_store()
    patient = store.create_patient(req.name)
    return {
        "patient": {
            **asdict(patient),
            "lesion_count": 0
        }
    }


@router.get("/{patient_id}")
def get_patient(patient_id: str):
    """Get a patient by ID"""
    store = get_case_store()
    patient = store.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    return {
        "patient": {
            **asdict(patient),
            "lesion_count": store.get_patient_lesion_count(patient_id)
        }
    }


@router.delete("/{patient_id}")
def delete_patient(patient_id: str):
    """Delete a patient and all their lesions"""
    store = get_case_store()
    patient = store.get_patient(patient_id)
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")

    store.delete_patient(patient_id)
    return {"success": True}
