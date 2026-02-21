"""
Sidebar Component - Shows previous queries for a patient
"""

import gradio as gr
from datetime import datetime
from typing import List, Optional
from data.case_store import get_case_store, Instance


def format_query_item(instance: Instance, index: int) -> str:
    """Format an instance as a query item for display"""
    diagnosis = "Pending"
    if instance.analysis and instance.analysis.get("diagnosis"):
        diag = instance.analysis["diagnosis"]
        diagnosis = diag.get("full_name", diag.get("class", "Unknown"))

    try:
        dt = datetime.fromisoformat(instance.created_at.replace('Z', '+00:00'))
        date_str = dt.strftime("%b %d, %H:%M")
    except:
        date_str = "Unknown"

    return f"Query #{index}: {diagnosis} ({date_str})"


def create_sidebar():
    """
    Create the sidebar component for showing previous queries.

    Returns:
        Tuple of (container, components dict)
    """
    with gr.Column(visible=False, elem_classes=["query-sidebar"]) as container:
        gr.Markdown("### Previous Queries", elem_classes=["sidebar-header"])

        # Dynamic list of query buttons
        query_list = gr.Column(elem_id="query-list")

        # New query button
        new_query_btn = gr.Button("+ New Query", size="sm", variant="primary")

    components = {
        "query_list": query_list,
        "new_query_btn": new_query_btn
    }

    return container, components


def get_queries_for_case(case_id: str) -> List[Instance]:
    """Get all instances/queries for a case"""
    if not case_id:
        return []
    return get_case_store().list_instances(case_id)
