"""
SkinProAI Frontend - Modular Gradio application
"""

import gradio as gr
from typing import Dict, Generator, Optional
from datetime import datetime
import sys
import os
import re
import base64

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.case_store import get_case_store
from frontend.components.styles import MAIN_CSS
from frontend.components.analysis_view import format_output


# =============================================================================
# CONFIG
# =============================================================================

class Config:
    APP_TITLE = "SkinProAI"
    SERVER_PORT = int(os.environ.get("GRADIO_SERVER_PORT", "7860"))
    HF_SPACES = os.environ.get("SPACE_ID") is not None


# =============================================================================
# AGENT
# =============================================================================

class AnalysisAgent:
    """Wrapper for the MedGemma analysis agent"""

    def __init__(self):
        self.model = None
        self.loaded = False

    def load(self):
        if self.loaded:
            return
        from models.medgemma_agent import MedGemmaAgent
        self.model = MedGemmaAgent(verbose=True)
        self.model.load_model()
        self.loaded = True

    def analyze(self, image_path: str, question: str = "") -> Generator[str, None, None]:
        if not self.loaded:
            yield "[STAGE:loading]Loading AI models...[/STAGE]\n"
            self.load()

        for chunk in self.model.analyze_image_stream(image_path, question=question):
            yield chunk

    def management_guidance(self, confirmed: bool, feedback: str = None) -> Generator[str, None, None]:
        for chunk in self.model.generate_management_guidance(confirmed, feedback):
            yield chunk

    def followup(self, message: str) -> Generator[str, None, None]:
        if not self.loaded or not self.model.last_diagnosis:
            yield "[ERROR]No analysis context available.[/ERROR]\n"
            return
        for chunk in self.model.chat_followup(message):
            yield chunk

    def reset(self):
        if self.model:
            self.model.reset_state()


agent = AnalysisAgent()
case_store = get_case_store()


# =============================================================================
# APP
# =============================================================================

with gr.Blocks(title=Config.APP_TITLE, css=MAIN_CSS, theme=gr.themes.Soft()) as app:

    # =========================================================================
    # STATE
    # =========================================================================
    state = gr.State({
        "page": "patient_select",  # patient_select | analysis
        "case_id": None,
        "instance_id": None,
        "output": "",
        "gradcam_base64": None
    })

    # =========================================================================
    # PAGE 1: PATIENT SELECTION
    # =========================================================================
    with gr.Group(visible=True, elem_classes=["patient-select-container"]) as page_patient:
        gr.Markdown("# SkinProAI", elem_classes=["patient-select-title"])
        gr.Markdown("Select a patient to continue or create a new case", elem_classes=["patient-select-subtitle"])

        with gr.Row(elem_classes=["patient-grid"]):
            btn_demo_melanoma = gr.Button("Demo: Melanocytic Lesion", elem_classes=["patient-card"])
            btn_demo_ak = gr.Button("Demo: Actinic Keratosis", elem_classes=["patient-card"])
            btn_new_patient = gr.Button("+ New Patient", variant="primary", elem_classes=["new-patient-btn"])

    # =========================================================================
    # PAGE 2: ANALYSIS
    # =========================================================================
    with gr.Group(visible=False) as page_analysis:

        # Header
        with gr.Row(elem_classes=["app-header"]):
            gr.Markdown(f"**{Config.APP_TITLE}**", elem_classes=["app-title"])
            btn_back = gr.Button("< Back to Patients", elem_classes=["back-btn"])

        with gr.Row(elem_classes=["analysis-container"]):

            # Sidebar (previous queries)
            with gr.Column(scale=0, min_width=260, visible=False, elem_classes=["query-sidebar"]) as sidebar:
                gr.Markdown("### Previous Queries", elem_classes=["sidebar-header"])
                sidebar_list = gr.Column(elem_id="sidebar-queries")
                btn_new_query = gr.Button("+ New Query", size="sm", variant="primary")

            # Main content
            with gr.Column(scale=4, elem_classes=["main-content"]):

                # Input view (greeting style)
                with gr.Group(visible=True, elem_classes=["input-greeting"]) as view_input:
                    gr.Markdown("What would you like to analyze?", elem_classes=["greeting-title"])
                    gr.Markdown("Upload an image and describe what you'd like to know", elem_classes=["greeting-subtitle"])

                    with gr.Column(elem_classes=["input-box-container"]):
                        input_message = gr.Textbox(
                            placeholder="Describe the lesion or ask a question...",
                            show_label=False,
                            lines=2,
                            elem_classes=["message-input"]
                        )

                        input_image = gr.Image(
                            type="pil",
                            height=180,
                            show_label=False,
                            elem_classes=["image-preview"]
                        )

                        with gr.Row(elem_classes=["input-actions"]):
                            gr.Markdown("*Upload a skin lesion image*")
                            btn_analyze = gr.Button("Analyze", elem_classes=["send-btn"], interactive=False)

                # Results view (shown after analysis)
                with gr.Group(visible=False, elem_classes=["chat-view"]) as view_results:
                    output_html = gr.HTML(
                        value='<div class="analysis-output">Starting...</div>',
                        elem_classes=["results-area"]
                    )

                    # Confirmation
                    with gr.Group(visible=False, elem_classes=["confirm-buttons"]) as confirm_box:
                        gr.Markdown("**Do you agree with this diagnosis?**")
                        with gr.Row():
                            btn_confirm_yes = gr.Button("Yes, continue", variant="primary", size="sm")
                            btn_confirm_no = gr.Button("No, I disagree", variant="secondary", size="sm")
                        input_feedback = gr.Textbox(label="Your assessment", placeholder="Enter diagnosis...", visible=False)
                        btn_submit_feedback = gr.Button("Submit", visible=False, size="sm")

                    # Follow-up
                    with gr.Row(elem_classes=["chat-input-area"]):
                        input_followup = gr.Textbox(placeholder="Ask a follow-up question...", show_label=False, lines=1, scale=4)
                        btn_followup = gr.Button("Send", size="sm", scale=1)

    # =========================================================================
    # DYNAMIC SIDEBAR RENDERING
    # =========================================================================
    @gr.render(inputs=[state], triggers=[state.change])
    def render_sidebar(s):
        case_id = s.get("case_id")
        if not case_id or s.get("page") != "analysis":
            return

        instances = case_store.list_instances(case_id)
        current = s.get("instance_id")

        for i, inst in enumerate(instances, 1):
            diagnosis = "Pending"
            if inst.analysis and inst.analysis.get("diagnosis"):
                d = inst.analysis["diagnosis"]
                diagnosis = d.get("class", "?")

            label = f"#{i}: {diagnosis}"
            variant = "primary" if inst.id == current else "secondary"
            btn = gr.Button(label, size="sm", variant=variant, elem_classes=["query-item"])

            # Attach click handler to load this instance
            def load_instance(inst_id=inst.id, c_id=case_id):
                def _load(current_state):
                    current_state["instance_id"] = inst_id
                    instance = case_store.get_instance(c_id, inst_id)

                    # Load saved output if available
                    output_html = '<div class="analysis-output"><div class="result">Previous analysis loaded</div></div>'
                    if instance and instance.analysis:
                        diag = instance.analysis.get("diagnosis", {})
                        output_html = f'<div class="analysis-output"><div class="result">Diagnosis: {diag.get("full_name", diag.get("class", "Unknown"))}</div></div>'

                    return (
                        current_state,
                        gr.update(visible=False),  # view_input
                        gr.update(visible=True),   # view_results
                        output_html,
                        gr.update(visible=False)   # confirm_box
                    )
                return _load

            btn.click(
                load_instance(),
                inputs=[state],
                outputs=[state, view_input, view_results, output_html, confirm_box]
            )

    # =========================================================================
    # EVENT HANDLERS
    # =========================================================================

    def select_patient(case_id: str, s: Dict):
        """Handle patient selection"""
        s["case_id"] = case_id
        s["page"] = "analysis"

        instances = case_store.list_instances(case_id)
        has_queries = len(instances) > 0

        if has_queries:
            # Load most recent
            inst = instances[-1]
            s["instance_id"] = inst.id

            # Load image if exists
            img = None
            if inst.image_path and os.path.exists(inst.image_path):
                from PIL import Image
                img = Image.open(inst.image_path)

            return (
                s,
                gr.update(visible=False),  # page_patient
                gr.update(visible=True),   # page_analysis
                gr.update(visible=True),   # sidebar
                gr.update(visible=False),  # view_input
                gr.update(visible=True),   # view_results
                '<div class="analysis-output"><div class="result">Previous analysis loaded</div></div>',
                gr.update(visible=False)   # confirm_box
            )
        else:
            # New instance
            inst = case_store.create_instance(case_id)
            s["instance_id"] = inst.id
            s["output"] = ""

            return (
                s,
                gr.update(visible=False),
                gr.update(visible=True),
                gr.update(visible=False),  # sidebar hidden for new patient
                gr.update(visible=True),   # view_input
                gr.update(visible=False),  # view_results
                "",
                gr.update(visible=False)
            )

    def new_patient(s: Dict):
        """Create new patient"""
        case = case_store.create_case(f"Patient {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        return select_patient(case.id, s)

    def go_back(s: Dict):
        """Return to patient selection"""
        s["page"] = "patient_select"
        s["case_id"] = None
        s["instance_id"] = None
        s["output"] = ""

        return (
            s,
            gr.update(visible=True),   # page_patient
            gr.update(visible=False),  # page_analysis
            gr.update(visible=False),  # sidebar
            gr.update(visible=True),   # view_input
            gr.update(visible=False),  # view_results
            "",
            gr.update(visible=False)   # confirm_box
        )

    def new_query(s: Dict):
        """Start new query for current patient"""
        case_id = s.get("case_id")
        if not case_id:
            return s, gr.update(), gr.update(), gr.update(), "", gr.update()

        inst = case_store.create_instance(case_id)
        s["instance_id"] = inst.id
        s["output"] = ""
        s["gradcam_base64"] = None

        agent.reset()

        return (
            s,
            gr.update(visible=True),   # view_input
            gr.update(visible=False),  # view_results
            None,                       # clear image
            "",                         # clear output
            gr.update(visible=False)   # confirm_box
        )

    def enable_analyze(img):
        """Enable analyze button when image uploaded"""
        return gr.update(interactive=img is not None)

    def run_analysis(image, message, s: Dict):
        """Run analysis on uploaded image"""
        if image is None:
            yield s, gr.update(), gr.update(), gr.update(), gr.update()
            return

        case_id = s["case_id"]
        instance_id = s["instance_id"]

        # Save image
        image_path = case_store.save_image(case_id, instance_id, image)
        case_store.update_analysis(case_id, instance_id, stage="analyzing", image_path=image_path)

        agent.reset()
        s["output"] = ""
        gradcam_base64 = None
        has_confirm = False

        # Switch to results view
        yield (
            s,
            gr.update(visible=False),  # view_input
            gr.update(visible=True),   # view_results
            '<div class="analysis-output">Starting analysis...</div>',
            gr.update(visible=False)   # confirm_box
        )

        partial = ""
        for chunk in agent.analyze(image_path, message or ""):
            partial += chunk

            # Check for GradCAM
            if gradcam_base64 is None:
                match = re.search(r'\[GRADCAM_IMAGE:([^\]]+)\]', partial)
                if match:
                    path = match.group(1)
                    if os.path.exists(path):
                        try:
                            with open(path, "rb") as f:
                                gradcam_base64 = base64.b64encode(f.read()).decode('utf-8')
                            s["gradcam_base64"] = gradcam_base64
                        except:
                            pass

            if '[CONFIRM:' in partial:
                has_confirm = True

            s["output"] = partial

            yield (
                s,
                gr.update(visible=False),
                gr.update(visible=True),
                format_output(partial, gradcam_base64),
                gr.update(visible=has_confirm)
            )

        # Save analysis
        if agent.model and agent.model.last_diagnosis:
            diag = agent.model.last_diagnosis["predictions"][0]
            case_store.update_analysis(
                case_id, instance_id,
                stage="awaiting_confirmation",
                analysis={"diagnosis": diag}
            )

    def confirm_yes(s: Dict):
        """User confirmed diagnosis"""
        partial = s.get("output", "")
        gradcam = s.get("gradcam_base64")

        for chunk in agent.management_guidance(confirmed=True):
            partial += chunk
            s["output"] = partial
            yield s, format_output(partial, gradcam), gr.update(visible=False)

        case_store.update_analysis(s["case_id"], s["instance_id"], stage="complete")

    def confirm_no():
        """Show feedback input"""
        return gr.update(visible=True), gr.update(visible=True)

    def submit_feedback(feedback: str, s: Dict):
        """Submit user feedback"""
        partial = s.get("output", "")
        gradcam = s.get("gradcam_base64")

        for chunk in agent.management_guidance(confirmed=False, feedback=feedback):
            partial += chunk
            s["output"] = partial
            yield (
                s,
                format_output(partial, gradcam),
                gr.update(visible=False),
                gr.update(visible=False),
                gr.update(visible=False),
                ""
            )

        case_store.update_analysis(s["case_id"], s["instance_id"], stage="complete")

    def send_followup(message: str, s: Dict):
        """Send follow-up question"""
        if not message.strip():
            return s, gr.update(), ""

        case_store.add_chat_message(s["case_id"], s["instance_id"], "user", message)

        partial = s.get("output", "")
        gradcam = s.get("gradcam_base64")

        partial += f'\n<div class="chat-message user">You: {message}</div>\n'

        response = ""
        for chunk in agent.followup(message):
            response += chunk
            s["output"] = partial + response
            yield s, format_output(partial + response, gradcam), ""

        case_store.add_chat_message(s["case_id"], s["instance_id"], "assistant", response)

    # =========================================================================
    # WIRE EVENTS
    # =========================================================================

    # Patient selection
    btn_demo_melanoma.click(
        lambda s: select_patient("demo-melanoma", s),
        inputs=[state],
        outputs=[state, page_patient, page_analysis, sidebar, view_input, view_results, output_html, confirm_box]
    )

    btn_demo_ak.click(
        lambda s: select_patient("demo-ak", s),
        inputs=[state],
        outputs=[state, page_patient, page_analysis, sidebar, view_input, view_results, output_html, confirm_box]
    )

    btn_new_patient.click(
        new_patient,
        inputs=[state],
        outputs=[state, page_patient, page_analysis, sidebar, view_input, view_results, output_html, confirm_box]
    )

    # Navigation
    btn_back.click(
        go_back,
        inputs=[state],
        outputs=[state, page_patient, page_analysis, sidebar, view_input, view_results, output_html, confirm_box]
    )

    btn_new_query.click(
        new_query,
        inputs=[state],
        outputs=[state, view_input, view_results, input_image, output_html, confirm_box]
    )

    # Analysis
    input_image.change(enable_analyze, inputs=[input_image], outputs=[btn_analyze])

    btn_analyze.click(
        run_analysis,
        inputs=[input_image, input_message, state],
        outputs=[state, view_input, view_results, output_html, confirm_box]
    )

    # Confirmation
    btn_confirm_yes.click(
        confirm_yes,
        inputs=[state],
        outputs=[state, output_html, confirm_box]
    )

    btn_confirm_no.click(
        confirm_no,
        outputs=[input_feedback, btn_submit_feedback]
    )

    btn_submit_feedback.click(
        submit_feedback,
        inputs=[input_feedback, state],
        outputs=[state, output_html, confirm_box, input_feedback, btn_submit_feedback, input_feedback]
    )

    # Follow-up
    btn_followup.click(
        send_followup,
        inputs=[input_followup, state],
        outputs=[state, output_html, input_followup]
    )

    input_followup.submit(
        send_followup,
        inputs=[input_followup, state],
        outputs=[state, output_html, input_followup]
    )


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print(f"\n{'='*50}")
    print(f"  {Config.APP_TITLE}")
    print(f"{'='*50}\n")

    app.queue().launch(
        server_name="0.0.0.0" if Config.HF_SPACES else "127.0.0.1",
        server_port=Config.SERVER_PORT,
        share=False,
        show_error=True
    )
