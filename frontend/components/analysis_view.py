"""
Analysis View Component - Main analysis interface with input and results
"""

import gradio as gr
import re
from typing import Optional


def parse_markdown(text: str) -> str:
    """Convert basic markdown to HTML"""
    text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
    text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
    text = re.sub(r'\*(.+?)\*', r'<em>\1</em>', text)

    # Bullet lists
    lines = text.split('\n')
    in_list = False
    result = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^[\*\-] ', stripped):
            if not in_list:
                result.append('<ul>')
                in_list = True
            item = re.sub(r'^[\*\-] ', '', stripped)
            result.append(f'<li>{item}</li>')
        else:
            if in_list:
                result.append('</ul>')
                in_list = False
            result.append(line)
    if in_list:
        result.append('</ul>')

    return '\n'.join(result)


# Regex patterns for output parsing
_STAGE_RE = re.compile(r'\[STAGE:(\w+)\](.*?)\[/STAGE\]')
_THINKING_RE = re.compile(r'\[THINKING\](.*?)\[/THINKING\]')
_OBSERVATION_RE = re.compile(r'\[OBSERVATION\](.*?)\[/OBSERVATION\]')
_TOOL_OUTPUT_RE = re.compile(r'\[TOOL_OUTPUT:(.*?)\]\n(.*?)\[/TOOL_OUTPUT\]', re.DOTALL)
_RESULT_RE = re.compile(r'\[RESULT\](.*?)\[/RESULT\]')
_ERROR_RE = re.compile(r'\[ERROR\](.*?)\[/ERROR\]')
_GRADCAM_RE = re.compile(r'\[GRADCAM_IMAGE:[^\]]+\]\n?')
_RESPONSE_RE = re.compile(r'\[RESPONSE\]\n(.*?)\n\[/RESPONSE\]', re.DOTALL)
_COMPLETE_RE = re.compile(r'\[COMPLETE\](.*?)\[/COMPLETE\]')
_CONFIRM_RE = re.compile(r'\[CONFIRM:(\w+)\](.*?)\[/CONFIRM\]')
_REFERENCES_RE = re.compile(r'\[REFERENCES\](.*?)\[/REFERENCES\]', re.DOTALL)
_REF_RE = re.compile(r'\[REF:([^:]+):([^:]+):([^:]+):([^:]+):([^\]]+)\]')


def format_output(raw_text: str, gradcam_base64: Optional[str] = None) -> str:
    """Convert tagged output to styled HTML"""
    html = raw_text

    # Stage headers
    html = _STAGE_RE.sub(
        r'<div class="stage"><span class="stage-indicator"></span><span class="stage-text">\2</span></div>',
        html
    )

    # Thinking
    html = _THINKING_RE.sub(r'<div class="thinking">\1</div>', html)

    # Observations
    html = _OBSERVATION_RE.sub(r'<div class="observation">\1</div>', html)

    # Tool outputs
    html = _TOOL_OUTPUT_RE.sub(
        r'<div class="tool-output"><div class="tool-header">\1</div><pre class="tool-content">\2</pre></div>',
        html
    )

    # Results
    html = _RESULT_RE.sub(r'<div class="result">\1</div>', html)

    # Errors
    html = _ERROR_RE.sub(r'<div class="error">\1</div>', html)

    # GradCAM image
    if gradcam_base64:
        img_html = f'<div class="gradcam-inline"><div class="gradcam-header">Attention Map</div><img src="data:image/png;base64,{gradcam_base64}" alt="Grad-CAM"></div>'
        html = _GRADCAM_RE.sub(img_html, html)
    else:
        html = _GRADCAM_RE.sub('', html)

    # Response section
    def format_response(match):
        content = match.group(1)
        parsed = parse_markdown(content)
        parsed = re.sub(r'\n\n+', '</p><p>', parsed)
        parsed = parsed.replace('\n', '<br>')
        return f'<div class="response"><p>{parsed}</p></div>'

    html = _RESPONSE_RE.sub(format_response, html)

    # Complete
    html = _COMPLETE_RE.sub(r'<div class="complete">\1</div>', html)

    # Confirmation
    html = _CONFIRM_RE.sub(
        r'<div class="confirm-box"><div class="confirm-text">\2</div></div>',
        html
    )

    # References
    def format_references(match):
        ref_content = match.group(1)
        refs_html = ['<div class="references"><div class="references-header">References</div><ul>']
        for ref_match in _REF_RE.finditer(ref_content):
            _, source, page, filename, superscript = ref_match.groups()
            refs_html.append(
                f'<li><a href="guidelines/{filename}#page={page}" target="_blank" class="ref-link">'
                f'<sup>{superscript}</sup> {source}, p.{page}</a></li>'
            )
        refs_html.append('</ul></div>')
        return '\n'.join(refs_html)

    html = _REFERENCES_RE.sub(format_references, html)

    # Convert newlines
    html = html.replace('\n', '<br>')

    return f'<div class="analysis-output">{html}</div>'


def create_analysis_view():
    """
    Create the analysis view component.

    Returns:
        Tuple of (container, components dict)
    """
    with gr.Group(visible=False, elem_classes=["analysis-container"]) as container:

        with gr.Row():
            # Main content area
            with gr.Column(elem_classes=["main-content"]):

                # Input greeting (shown when no analysis yet)
                with gr.Group(visible=True, elem_classes=["input-greeting"]) as input_greeting:
                    gr.Markdown("What would you like to analyze?", elem_classes=["greeting-title"])
                    gr.Markdown("Upload an image and describe what you'd like to know", elem_classes=["greeting-subtitle"])

                    with gr.Column(elem_classes=["input-box-container"]):
                        message_input = gr.Textbox(
                            placeholder="Describe the lesion or ask a question...",
                            show_label=False,
                            lines=3,
                            elem_classes=["message-input"]
                        )

                        # Image upload (compact)
                        image_input = gr.Image(
                            label="",
                            type="pil",
                            height=180,
                            elem_classes=["image-preview"],
                            show_label=False
                        )

                        with gr.Row(elem_classes=["input-actions"]):
                            upload_hint = gr.Markdown("*Upload a skin lesion image above*", visible=True)
                            send_btn = gr.Button("Analyze", elem_classes=["send-btn"], interactive=False)

                # Chat/results view (shown after analysis starts)
                with gr.Group(visible=False, elem_classes=["chat-view"]) as chat_view:
                    results_output = gr.HTML(
                        value='<div class="analysis-output">Starting analysis...</div>',
                        elem_classes=["results-area"]
                    )

                    # Confirmation buttons
                    with gr.Group(visible=False, elem_classes=["confirm-buttons"]) as confirm_group:
                        gr.Markdown("**Do you agree with this diagnosis?**")
                        with gr.Row():
                            confirm_yes_btn = gr.Button("Yes, continue", variant="primary", size="sm")
                            confirm_no_btn = gr.Button("No, I disagree", variant="secondary", size="sm")
                        feedback_input = gr.Textbox(
                            label="Your assessment",
                            placeholder="Enter your diagnosis...",
                            visible=False
                        )
                        submit_feedback_btn = gr.Button("Submit", visible=False, size="sm")

                    # Follow-up input
                    with gr.Row(elem_classes=["chat-input-area"]):
                        followup_input = gr.Textbox(
                            placeholder="Ask a follow-up question...",
                            show_label=False,
                            lines=1
                        )
                        followup_btn = gr.Button("Send", size="sm", elem_classes=["send-btn"])

    components = {
        "input_greeting": input_greeting,
        "chat_view": chat_view,
        "message_input": message_input,
        "image_input": image_input,
        "send_btn": send_btn,
        "results_output": results_output,
        "confirm_group": confirm_group,
        "confirm_yes_btn": confirm_yes_btn,
        "confirm_no_btn": confirm_no_btn,
        "feedback_input": feedback_input,
        "submit_feedback_btn": submit_feedback_btn,
        "followup_input": followup_input,
        "followup_btn": followup_btn,
        "upload_hint": upload_hint
    }

    return container, components
