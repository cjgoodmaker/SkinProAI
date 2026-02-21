"""
Patient Selection Component - Landing page for selecting/creating patients
"""

import gradio as gr
from typing import Callable, List
from data.case_store import get_case_store, Case


def create_patient_select(on_patient_selected: Callable[[str], None]) -> gr.Group:
    """
    Create the patient selection page component.

    Args:
        on_patient_selected: Callback when a patient is selected (receives case_id)

    Returns:
        gr.Group containing the patient selection UI
    """
    case_store = get_case_store()

    with gr.Group(visible=True, elem_classes=["patient-select-container"]) as container:
        gr.Markdown("# SkinProAI", elem_classes=["patient-select-title"])
        gr.Markdown("Select a patient to continue or create a new case", elem_classes=["patient-select-subtitle"])

        with gr.Column(elem_classes=["patient-grid"]):
            # Demo cases
            demo_melanoma_btn = gr.Button(
                "Demo: Melanocytic Lesion",
                elem_classes=["patient-card"]
            )
            demo_ak_btn = gr.Button(
                "Demo: Actinic Keratosis",
                elem_classes=["patient-card"]
            )

            # New patient button
            new_patient_btn = gr.Button(
                "+ New Patient",
                elem_classes=["new-patient-btn"]
            )

    return container, demo_melanoma_btn, demo_ak_btn, new_patient_btn


def get_patient_cases() -> List[Case]:
    """Get list of all patient cases"""
    return get_case_store().list_cases()
