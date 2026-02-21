"""
MedGemma Agent - LLM agent with tool calling and staged thinking feedback

Pipeline: MedGemma independent exam → Tools (MONET/ConvNeXt/GradCAM) → MedGemma reconciliation → Management
"""

import sys
import time
import random
import json
import os
import subprocess
import threading
from typing import Optional, Generator, Dict, Any
from PIL import Image


class MCPClient:
    """
    Minimal MCP client that communicates with a FastMCP subprocess over stdio.

    Uses raw newline-delimited JSON-RPC 2.0 so the main process (Python 3.9)
    does not need the mcp library. The subprocess is launched with python3.11
    which has mcp installed.
    """

    def __init__(self):
        self._process = None
        self._lock = threading.Lock()
        self._id_counter = 0

    def _next_id(self) -> int:
        self._id_counter += 1
        return self._id_counter

    def _send(self, obj: dict):
        line = json.dumps(obj) + "\n"
        self._process.stdin.write(line)
        self._process.stdin.flush()

    def _recv(self) -> dict:
        while True:
            line = self._process.stdout.readline()
            if not line:
                raise RuntimeError("MCP server closed connection unexpectedly")
            line = line.strip()
            if not line:
                continue
            msg = json.loads(line)
            # Skip server-initiated notifications (no "id" key)
            if "id" in msg:
                return msg

    def _initialize(self):
        """Send MCP initialize handshake."""
        req_id = self._next_id()
        self._send({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": "initialize",
            "params": {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "SkinProAI", "version": "1.0.0"},
            },
        })
        self._recv()  # consume initialize response
        # Confirm initialization
        self._send({
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        })

    def start(self):
        """Spawn the MCP server subprocess and complete the handshake."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        server_script = os.path.join(root, "mcp_server", "server.py")
        self._process = subprocess.Popen(
            [sys.executable, server_script],  # use same venv Python (has all ML packages)
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,  # line-buffered
        )
        self._initialize()

    def call_tool_sync(self, tool_name: str, arguments: dict) -> dict:
        """Call a tool synchronously and return the parsed result dict."""
        with self._lock:
            req_id = self._next_id()
            self._send({
                "jsonrpc": "2.0",
                "id": req_id,
                "method": "tools/call",
                "params": {"name": tool_name, "arguments": arguments},
            })
            response = self._recv()

        # Protocol-level error (e.g. unknown method)
        if "error" in response:
            raise RuntimeError(
                f"MCP tool '{tool_name}' failed: {response['error']}"
            )

        result = response["result"]
        content_text = result["content"][0]["text"]

        # Tool-level error (isError=True means the tool itself raised an exception)
        if result.get("isError"):
            raise RuntimeError(f"MCP tool '{tool_name}' error: {content_text}")

        return json.loads(content_text)

    def stop(self):
        """Terminate the MCP server subprocess."""
        if self._process:
            try:
                self._process.stdin.close()
                self._process.terminate()
                self._process.wait(timeout=5)
            except Exception:
                pass
            self._process = None


# Rotating verbs for spinner effect
ANALYSIS_VERBS = [
    "Analyzing", "Examining", "Processing", "Inspecting", "Evaluating",
    "Scanning", "Assessing", "Reviewing", "Studying", "Interpreting"
]

# Comprehensive visual exam prompt (combined from 4 separate stages)
COMPREHENSIVE_EXAM_PROMPT = """Perform a systematic dermoscopic examination of this skin lesion. Assess ALL of the following in a SINGLE concise analysis:

1. PATTERN: Overall architecture, symmetry (symmetric/asymmetric), organization
2. COLORS: List all colors present (brown, black, blue, white, red, pink) and distribution
3. BORDER: Sharp vs gradual, regular vs irregular, any disruptions
4. STRUCTURES: Pigment network, dots/globules, streaks, blue-white veil, regression, vessels

Then provide:
- Top 3 differential diagnoses with brief reasoning
- Concern level (1-5, where 5=urgent)
- Single most important feature driving your assessment

Be CONCISE - focus on clinically relevant findings only."""


def get_verb():
    """Get a random analysis verb for spinner effect"""
    return random.choice(ANALYSIS_VERBS)


class MedGemmaAgent:
    """
    Medical image analysis agent with:
    - Staged thinking display (no emojis)
    - Tool calling (MONET, ConvNeXt, Grad-CAM)
    - Streaming responses
    """

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.pipe = None
        self.model_id = "google/medgemma-4b-it"
        self.loaded = False

        # Tools (legacy direct instances, kept for fallback / non-MCP use)
        self.monet_tool = None
        self.convnext_tool = None
        self.gradcam_tool = None
        self.rag_tool = None
        self.tools_loaded = False

        # MCP client
        self.mcp_client = None

        # State for confirmation flow
        self.last_diagnosis = None
        self.last_monet_result = None
        self.last_image = None
        self.last_medgemma_exam = None  # Store independent MedGemma findings
        self.last_reconciliation = None

    def reset_state(self):
        """Reset analysis state for new analysis (keeps models loaded)"""
        self.last_diagnosis = None
        self.last_monet_result = None
        self.last_image = None
        self.last_medgemma_exam = None
        self.last_reconciliation = None

    def _print(self, message: str):
        """Print if verbose"""
        if self.verbose:
            print(message, flush=True)

    def load_model(self):
        """Load MedGemma model"""
        if self.loaded:
            return

        self._print("Initializing MedGemma agent...")

        import os
        import torch
        from transformers import pipeline

        # Authenticate with HF Hub (required for gated models like MedGemma)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token, add_to_git_credential=False)
            self._print("Authenticated with HF Hub")
        else:
            self._print("Warning: HF_TOKEN not set — gated models will fail")

        self._print(f"Loading model: {self.model_id}")

        if torch.cuda.is_available():
            device = "cuda"
            self._print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        elif torch.backends.mps.is_available():
            device = "mps"
            self._print("Using Apple Silicon (MPS)")
        else:
            device = "cpu"
            self._print("Using CPU")

        model_kwargs = dict(
            torch_dtype=torch.bfloat16 if device != "cpu" else torch.float32,
            device_map="auto",
        )

        start = time.time()
        self.pipe = pipeline(
            "image-text-to-text",
            model=self.model_id,
            model_kwargs=model_kwargs,
            token=hf_token,  # pass explicitly in addition to login()
        )

        self._print(f"Model loaded in {time.time() - start:.1f}s")
        self.loaded = True

    def load_tools(self):
        """Load tool models (MONET + ConvNeXt + Grad-CAM + RAG)"""
        if self.tools_loaded:
            return

        from models.monet_tool import MonetTool
        self.monet_tool = MonetTool()
        self.monet_tool.load()

        from models.convnext_classifier import ConvNeXtClassifier
        self.convnext_tool = ConvNeXtClassifier()
        self.convnext_tool.load()

        from models.gradcam_tool import GradCAMTool
        self.gradcam_tool = GradCAMTool(classifier=self.convnext_tool)
        self.gradcam_tool.load()

        from models.guidelines_rag import get_guidelines_rag
        self.rag_tool = get_guidelines_rag()
        if not self.rag_tool.loaded:
            self.rag_tool.load_index()

        self.tools_loaded = True

    def load_tools_via_mcp(self):
        """Start the MCP server subprocess and mark tools as loaded."""
        if self.tools_loaded:
            return
        self.mcp_client = MCPClient()
        self.mcp_client.start()
        self.tools_loaded = True

    def _multi_pass_visual_exam(self, image, question: Optional[str] = None) -> Generator[str, None, Dict[str, str]]:
        """
        MedGemma performs comprehensive visual examination BEFORE tools run.
        Single prompt covers pattern, colors, borders, structures, and differentials.
        Returns findings dict after yielding all output.
        """
        findings = {}

        yield f"\n[STAGE:medgemma_exam]MedGemma Visual Examination[/STAGE]\n"
        yield f"[THINKING]Performing systematic dermoscopic assessment...[/THINKING]\n"

        # Build prompt with optional clinical question
        exam_prompt = COMPREHENSIVE_EXAM_PROMPT
        if question:
            exam_prompt += f"\n\nCLINICAL QUESTION: {question}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": exam_prompt}
                ]
            }
        ]

        try:
            time.sleep(0.2)
            output = self.pipe(messages, max_new_tokens=400)
            result = output[0]["generated_text"][-1]["content"]
            findings['synthesis'] = result

            yield f"[RESPONSE]\n"
            words = result.split()
            for i, word in enumerate(words):
                time.sleep(0.015)
                yield word + (" " if i < len(words) - 1 else "")
            yield f"\n[/RESPONSE]\n"

        except Exception as e:
            findings['synthesis'] = f"Analysis failed: {e}"
            yield f"[ERROR]Visual examination failed: {e}[/ERROR]\n"

        self.last_medgemma_exam = findings
        return findings

    def _reconcile_findings(
        self,
        image,
        medgemma_exam: Dict[str, str],
        monet_result: Dict[str, Any],
        convnext_result: Dict[str, Any],
        question: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        MedGemma reconciles its independent findings with tool outputs.
        Identifies agreements, disagreements, and provides integrated assessment.
        """
        yield f"\n[STAGE:reconciliation]Reconciling MedGemma Findings with Tool Results[/STAGE]\n"
        yield f"[THINKING]Comparing independent visual assessment against AI classification tools...[/THINKING]\n"

        top = convnext_result['predictions'][0]
        runner_up = convnext_result['predictions'][1] if len(convnext_result['predictions']) > 1 else None

        # Build MONET features string
        monet_top = sorted(monet_result["features"].items(), key=lambda x: x[1], reverse=True)[:5]
        monet_str = ", ".join([f"{k.replace('MONET_', '').replace('_', ' ')}: {v:.0%}" for k, v in monet_top])

        reconciliation_prompt = f"""You performed an independent visual examination of this lesion and concluded:

YOUR ASSESSMENT:
{medgemma_exam.get('synthesis', 'Not available')[:600]}

The AI classification tools produced these results:
- ConvNeXt classifier: {top['full_name']} ({top['probability']:.1%} confidence)
{f"- Runner-up: {runner_up['full_name']} ({runner_up['probability']:.1%})" if runner_up else ""}
- Key MONET features: {monet_str}

{f'CLINICAL QUESTION: {question}' if question else ''}

Reconcile your visual findings with the AI classification:
1. AGREEMENT/DISAGREEMENT: Do your findings support the AI diagnosis? Any conflicts?
2. INTEGRATED ASSESSMENT: Final diagnosis considering all evidence
3. CONFIDENCE (1-10): How certain? What would change your assessment?

Be concise and specific."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": reconciliation_prompt}
                ]
            }
        ]

        try:
            output = self.pipe(messages, max_new_tokens=300)
            reconciliation = output[0]["generated_text"][-1]["content"]
            self.last_reconciliation = reconciliation

            yield f"[RESPONSE]\n"
            words = reconciliation.split()
            for i, word in enumerate(words):
                time.sleep(0.015)
                yield word + (" " if i < len(words) - 1 else "")
            yield f"\n[/RESPONSE]\n"

        except Exception as e:
            yield f"[ERROR]Reconciliation failed: {e}[/ERROR]\n"

    def analyze_image_stream(
        self,
        image_path: str,
        question: Optional[str] = None,
        max_tokens: int = 512,
        use_tools: bool = True
    ) -> Generator[str, None, None]:
        """
        Stream analysis with new pipeline:
        1. MedGemma independent multi-pass exam
        2. MONET + ConvNeXt + GradCAM tools
        3. MedGemma reconciliation
        4. Confirmation request
        """
        if not self.loaded:
            yield "[STAGE:loading]Initializing MedGemma...[/STAGE]\n"
            self.load_model()

        yield f"[STAGE:image]{get_verb()} image...[/STAGE]\n"

        try:
            image = Image.open(image_path).convert("RGB")
            self.last_image = image
        except Exception as e:
            yield f"[ERROR]Failed to load image: {e}[/ERROR]\n"
            return

        # Load tools early via MCP subprocess
        if use_tools and not self.tools_loaded:
            yield f"[STAGE:tools]Loading analysis tools...[/STAGE]\n"
            self.load_tools_via_mcp()

        # ===== PHASE 1: MedGemma Independent Visual Examination =====
        medgemma_exam = {}
        for chunk in self._multi_pass_visual_exam(image, question):
            yield chunk
            if isinstance(chunk, dict):
                medgemma_exam = chunk
        medgemma_exam = self.last_medgemma_exam or {}

        monet_result = None
        convnext_result = None

        if use_tools:
            # ===== PHASE 2: Run Classification Tools =====
            yield f"\n[STAGE:tools_run]Running AI Classification Tools[/STAGE]\n"
            yield f"[THINKING]Now running MONET and ConvNeXt to compare against visual examination...[/THINKING]\n"

            # MONET Feature Extraction
            time.sleep(0.2)
            yield f"\n[STAGE:monet]MONET Feature Extraction[/STAGE]\n"

            try:
                monet_result = self.mcp_client.call_tool_sync(
                    "monet_analyze", {"image_path": image_path}
                )
                self.last_monet_result = monet_result

                yield f"[TOOL_OUTPUT:MONET Features]\n"
                for name, score in monet_result["features"].items():
                    short_name = name.replace("MONET_", "").replace("_", " ").title()
                    bar_filled = int(score * 10)
                    bar = "|" + "=" * bar_filled + "-" * (10 - bar_filled) + "|"
                    yield f"  {short_name}: {bar} {score:.0%}\n"
                yield f"[/TOOL_OUTPUT]\n"

            except Exception as e:
                yield f"[ERROR]MONET failed: {e}[/ERROR]\n"

            # ConvNeXt Classification
            time.sleep(0.2)
            yield f"\n[STAGE:convnext]ConvNeXt Classification[/STAGE]\n"

            try:
                monet_scores = monet_result["vector"] if monet_result else None
                convnext_result = self.mcp_client.call_tool_sync(
                    "classify_lesion",
                    {
                        "image_path": image_path,
                        "monet_scores": monet_scores,
                    },
                )
                self.last_diagnosis = convnext_result

                yield f"[TOOL_OUTPUT:Classification Results]\n"
                for pred in convnext_result["predictions"][:5]:
                    prob = pred['probability']
                    bar_filled = int(prob * 20)
                    bar = "|" + "=" * bar_filled + "-" * (20 - bar_filled) + "|"
                    yield f"  {pred['class']}: {bar} {prob:.1%}\n"
                    yield f"    {pred['full_name']}\n"
                yield f"[/TOOL_OUTPUT]\n"

                top = convnext_result['predictions'][0]
                yield f"[RESULT]ConvNeXt Primary: {top['full_name']} ({top['probability']:.1%})[/RESULT]\n"

            except Exception as e:
                yield f"[ERROR]ConvNeXt failed: {e}[/ERROR]\n"

            # Grad-CAM Visualization
            time.sleep(0.2)
            yield f"\n[STAGE:gradcam]Grad-CAM Attention Map[/STAGE]\n"

            try:
                gradcam_result = self.mcp_client.call_tool_sync(
                    "generate_gradcam", {"image_path": image_path}
                )
                gradcam_path = gradcam_result["gradcam_path"]
                yield f"[GRADCAM_IMAGE:{gradcam_path}]\n"
            except Exception as e:
                yield f"[ERROR]Grad-CAM failed: {e}[/ERROR]\n"

            # ===== PHASE 3: MedGemma Reconciliation =====
            if convnext_result and monet_result and medgemma_exam:
                for chunk in self._reconcile_findings(
                    image, medgemma_exam, monet_result, convnext_result, question
                ):
                    yield chunk

        # Yield confirmation request
        if convnext_result:
            top = convnext_result['predictions'][0]
            yield f"\n[CONFIRM:diagnosis]Do you agree with the integrated assessment?[/CONFIRM]\n"

    def generate_management_guidance(
        self,
        user_confirmed: bool = True,
        user_feedback: Optional[str] = None
    ) -> Generator[str, None, None]:
        """
        Generate LESION-SPECIFIC management guidance using RAG + MedGemma reasoning.
        References specific findings from this analysis, not generic textbook management.
        """
        if not self.last_diagnosis:
            yield "[ERROR]No diagnosis available. Please analyze an image first.[/ERROR]\n"
            return

        top = self.last_diagnosis['predictions'][0]
        runner_up = self.last_diagnosis['predictions'][1] if len(self.last_diagnosis['predictions']) > 1 else None
        diagnosis = top['full_name']

        if not user_confirmed and user_feedback:
            yield f"[THINKING]Clinician provided alternative assessment: {user_feedback}[/THINKING]\n"
            diagnosis = user_feedback

        # Stage: RAG Search
        time.sleep(0.3)
        yield f"\n[STAGE:guidelines]Searching clinical guidelines for {diagnosis}...[/STAGE]\n"

        # Get RAG context via MCP
        features_desc = self.last_monet_result.get('description', '') if self.last_monet_result else ''
        rag_data = self.mcp_client.call_tool_sync(
            "search_guidelines",
            {"query": features_desc, "diagnosis": diagnosis},
        )
        context = rag_data["context"]
        references = rag_data["references"]

        # Check guideline relevance
        has_relevant_guidelines = False
        if references:
            diagnosis_lower = diagnosis.lower()
            for ref in references:
                source_lower = ref['source'].lower()
                if any(term in diagnosis_lower for term in ['melanoma']) and 'melanoma' in source_lower:
                    has_relevant_guidelines = True
                    break
                elif 'actinic' in diagnosis_lower and 'actinic' in source_lower:
                    has_relevant_guidelines = True
                    break
                elif ref.get('score', 0) > 0.7:
                    has_relevant_guidelines = True
                    break

        if not references or not has_relevant_guidelines:
            yield f"[THINKING]No specific published guidelines for {diagnosis}. Using clinical knowledge.[/THINKING]\n"
            context = "No specific clinical guidelines available."
            references = []

        # Build MONET features for context
        monet_features = ""
        if self.last_monet_result:
            top_features = sorted(self.last_monet_result["features"].items(), key=lambda x: x[1], reverse=True)[:5]
            monet_features = ", ".join([f"{k.replace('MONET_', '').replace('_', ' ')}: {v:.0%}" for k, v in top_features])

        # Stage: Lesion-Specific Management Reasoning
        time.sleep(0.3)
        yield f"\n[STAGE:management]Generating Lesion-Specific Management Plan[/STAGE]\n"
        yield f"[THINKING]Creating management plan tailored to THIS lesion's specific characteristics...[/THINKING]\n"

        management_prompt = f"""Generate a CONCISE management plan for this lesion:

DIAGNOSIS: {diagnosis} ({top['probability']:.1%})
{f"Alternative: {runner_up['full_name']} ({runner_up['probability']:.1%})" if runner_up else ""}
KEY FEATURES: {monet_features}

{f"GUIDELINES: {context[:800]}" if context else ""}

Provide:
1. RECOMMENDED ACTION: Biopsy, excision, monitoring, or discharge - with specific reasoning
2. URGENCY: Routine vs urgent vs same-day referral
3. KEY CONCERNS: What features drive this recommendation

Be specific to THIS lesion. 3-5 sentences maximum."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.last_image},
                    {"type": "text", "text": management_prompt}
                ]
            }
        ]

        # Generate response
        start = time.time()
        try:
            output = self.pipe(messages, max_new_tokens=250)
            response = output[0]["generated_text"][-1]["content"]

            yield f"[RESPONSE]\n"
            words = response.split()
            for i, word in enumerate(words):
                time.sleep(0.015)
                yield word + (" " if i < len(words) - 1 else "")
            yield f"\n[/RESPONSE]\n"

        except Exception as e:
            yield f"[ERROR]Management generation failed: {e}[/ERROR]\n"

        # Output references (pre-formatted by MCP server)
        if references:
            yield rag_data["references_display"]

        yield f"\n[COMPLETE]Lesion-specific management plan generated in {time.time() - start:.1f}s[/COMPLETE]\n"

        # Store response for recommendation extraction
        self.last_management_response = response

    def extract_recommendation(self) -> Generator[str, None, Dict[str, Any]]:
        """
        Extract structured recommendation from management guidance.
        Determines: BIOPSY, EXCISION, FOLLOWUP, or DISCHARGE
        For BIOPSY/EXCISION, gets coordinates from MedGemma.
        """
        if not self.last_management_response or not self.last_image:
            yield "[ERROR]No management guidance available[/ERROR]\n"
            return {"action": "UNKNOWN"}

        yield f"\n[STAGE:recommendation]Extracting Clinical Recommendation[/STAGE]\n"

        # Ask MedGemma to classify the recommendation
        classification_prompt = f"""Based on the management plan you just provided:

{self.last_management_response[:1000]}

Classify the PRIMARY recommended action into exactly ONE of these categories:
- BIOPSY: If punch biopsy, shave biopsy, or incisional biopsy is recommended
- EXCISION: If complete surgical excision is recommended
- FOLLOWUP: If monitoring with repeat photography/dermoscopy is recommended
- DISCHARGE: If the lesion is clearly benign and no follow-up needed

Respond with ONLY the category name (BIOPSY, EXCISION, FOLLOWUP, or DISCHARGE) on the first line.
Then on the second line, provide a brief (1 sentence) justification."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.last_image},
                    {"type": "text", "text": classification_prompt}
                ]
            }
        ]

        try:
            output = self.pipe(messages, max_new_tokens=100)
            response = output[0]["generated_text"][-1]["content"].strip()
            lines = response.split('\n')
            action = lines[0].strip().upper()
            justification = lines[1].strip() if len(lines) > 1 else ""

            # Validate action
            valid_actions = ["BIOPSY", "EXCISION", "FOLLOWUP", "DISCHARGE"]
            if action not in valid_actions:
                # Try to extract from response
                for valid in valid_actions:
                    if valid in response.upper():
                        action = valid
                        break
                else:
                    action = "FOLLOWUP"  # Default to safe option

            yield f"[RESULT]Recommended Action: {action}[/RESULT]\n"
            yield f"[OBSERVATION]{justification}[/OBSERVATION]\n"

            result = {
                "action": action,
                "justification": justification
            }

            return result

        except Exception as e:
            yield f"[ERROR]Failed to extract recommendation: {e}[/ERROR]\n"
            return {"action": "UNKNOWN", "error": str(e)}

    def compare_followup_images(
        self,
        previous_image_path: str,
        current_image_path: str
    ) -> Generator[str, None, None]:
        """
        Compare a follow-up image with the previous one.
        Runs full analysis pipeline on current image, then compares findings.
        """
        yield f"\n[STAGE:comparison]Follow-up Comparison Analysis[/STAGE]\n"

        try:
            current_image = Image.open(current_image_path).convert("RGB")
        except Exception as e:
            yield f"[ERROR]Failed to load images: {e}[/ERROR]\n"
            return

        # Store previous analysis state
        prev_exam = self.last_medgemma_exam

        # Generate comparison image and MONET deltas via MCP
        yield f"\n[STAGE:current_analysis]Analyzing Current Image[/STAGE]\n"

        if self.tools_loaded:
            try:
                compare_data = self.mcp_client.call_tool_sync(
                    "compare_images",
                    {
                        "image1_path": previous_image_path,
                        "image2_path": current_image_path,
                    },
                )
                yield f"[COMPARISON_IMAGE:{compare_data['comparison_path']}]\n"

                # Side-by-side GradCAM comparison if both paths available
                prev_gc = compare_data.get("prev_gradcam_path")
                curr_gc = compare_data.get("curr_gradcam_path")
                if prev_gc and curr_gc:
                    yield f"[GRADCAM_COMPARE:{prev_gc}:{curr_gc}]\n"

                # Display MONET feature deltas
                if compare_data["monet_deltas"]:
                    yield f"[TOOL_OUTPUT:Feature Comparison]\n"
                    for name, delta_info in compare_data["monet_deltas"].items():
                        prev_val = delta_info["previous"]
                        curr_val = delta_info["current"]
                        diff = delta_info["delta"]
                        short_name = name.replace("MONET_", "").replace("_", " ").title()
                        direction = "↑" if diff > 0 else "↓"
                        yield f"  {short_name}: {prev_val:.0%} → {curr_val:.0%} ({direction}{abs(diff):.0%})\n"
                    yield f"[/TOOL_OUTPUT]\n"

            except Exception as e:
                yield f"[ERROR]MCP comparison failed: {e}[/ERROR]\n"

        # MedGemma comparison analysis
        comparison_prompt = f"""You are comparing TWO images of the same skin lesion taken at different times.

PREVIOUS ANALYSIS:
{prev_exam.get('synthesis', 'Not available')[:500] if prev_exam else 'Not available'}

Now examine the CURRENT image and compare to your memory of the previous findings.

Assess for changes in:
1. SIZE: Has the lesion grown, shrunk, or stayed the same?
2. COLOR: Any new colors appeared? Any colors faded?
3. SHAPE/SYMMETRY: Has the shape changed? More or less symmetric?
4. BORDERS: Sharper, more irregular, or unchanged?
5. STRUCTURES: New dermoscopic structures? Lost structures?

Provide your assessment:
- CHANGE_LEVEL: SIGNIFICANT_CHANGE / MINOR_CHANGE / STABLE / IMPROVED
- Specific changes observed
- Clinical recommendation based on changes"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": current_image},
                    {"type": "text", "text": comparison_prompt}
                ]
            }
        ]

        try:
            yield f"[THINKING]Comparing current image to previous findings...[/THINKING]\n"
            output = self.pipe(messages, max_new_tokens=400)
            comparison_result = output[0]["generated_text"][-1]["content"]

            yield f"[RESPONSE]\n"
            words = comparison_result.split()
            for i, word in enumerate(words):
                time.sleep(0.02)
                yield word + (" " if i < len(words) - 1 else "")
            yield f"\n[/RESPONSE]\n"

            # Extract change level
            change_level = "UNKNOWN"
            for level in ["SIGNIFICANT_CHANGE", "MINOR_CHANGE", "STABLE", "IMPROVED"]:
                if level in comparison_result.upper():
                    change_level = level
                    break

            if change_level == "SIGNIFICANT_CHANGE":
                yield f"[RESULT]⚠️ SIGNIFICANT CHANGES DETECTED - Further evaluation recommended[/RESULT]\n"
            elif change_level == "IMPROVED":
                yield f"[RESULT]✓ LESION IMPROVED - Continue monitoring[/RESULT]\n"
            elif change_level == "STABLE":
                yield f"[RESULT]✓ LESION STABLE - Continue scheduled follow-up[/RESULT]\n"
            else:
                yield f"[RESULT]Minor changes noted - Clinical correlation recommended[/RESULT]\n"

        except Exception as e:
            yield f"[ERROR]Comparison analysis failed: {e}[/ERROR]\n"

        yield f"\n[COMPLETE]Follow-up comparison complete[/COMPLETE]\n"

    def chat(self, message: str, image_path: Optional[str] = None) -> str:
        """Simple chat interface"""
        if not self.loaded:
            self.load_model()

        content = []
        if image_path:
            image = Image.open(image_path).convert("RGB")
            content.append({"type": "image", "image": image})
        content.append({"type": "text", "text": message})

        messages = [{"role": "user", "content": content}]
        output = self.pipe(messages, max_new_tokens=512)
        return output[0]["generated_text"][-1]["content"]

    def chat_followup(self, message: str) -> Generator[str, None, None]:
        """
        Handle follow-up questions using the stored analysis context.
        Uses the last analyzed image and diagnosis to provide contextual responses.
        """
        if not self.loaded:
            yield "[ERROR]Model not loaded[/ERROR]\n"
            return

        if not self.last_diagnosis or not self.last_image:
            yield "[ERROR]No previous analysis context. Please analyze an image first.[/ERROR]\n"
            return

        # Build context from previous analysis
        top_diagnosis = self.last_diagnosis['predictions'][0]
        differentials = ", ".join([
            f"{p['class']} ({p['probability']:.0%})"
            for p in self.last_diagnosis['predictions'][:3]
        ])

        monet_desc = ""
        if self.last_monet_result:
            monet_desc = self.last_monet_result.get('description', '')

        context_prompt = f"""You are a dermatology assistant helping with skin lesion analysis.

PREVIOUS ANALYSIS CONTEXT:
- Primary diagnosis: {top_diagnosis['full_name']} ({top_diagnosis['probability']:.1%} confidence)
- Differential diagnoses: {differentials}
- Visual features: {monet_desc}

The user has a follow-up question about this lesion. Please provide a helpful, medically accurate response.

USER QUESTION: {message}

Provide a concise, informative response. If the question is outside your expertise or requires in-person examination, say so."""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": self.last_image},
                    {"type": "text", "text": context_prompt}
                ]
            }
        ]

        try:
            yield f"[THINKING]Considering your question in context of the previous analysis...[/THINKING]\n"
            time.sleep(0.2)

            output = self.pipe(messages, max_new_tokens=400)
            response = output[0]["generated_text"][-1]["content"]

            yield f"[RESPONSE]\n"
            # Stream word by word for typewriter effect
            words = response.split()
            for i, word in enumerate(words):
                time.sleep(0.02)
                yield word + (" " if i < len(words) - 1 else "")
            yield f"\n[/RESPONSE]\n"

        except Exception as e:
            yield f"[ERROR]Failed to generate response: {e}[/ERROR]\n"


def main():
    """Interactive terminal interface"""
    print("=" * 60)
    print("  MedGemma Agent - Medical Image Analysis")
    print("=" * 60)

    agent = MedGemmaAgent(verbose=True)
    agent.load_model()

    print("\nCommands: analyze <path>, chat <message>, quit")

    while True:
        try:
            user_input = input("\n> ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["quit", "exit", "q"]:
                break

            parts = user_input.split(maxsplit=1)
            cmd = parts[0].lower()

            if cmd == "analyze" and len(parts) > 1:
                for chunk in agent.analyze_image_stream(parts[1].strip()):
                    print(chunk, end="", flush=True)

            elif cmd == "chat" and len(parts) > 1:
                print(agent.chat(parts[1]))

            else:
                print("Unknown command")

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
