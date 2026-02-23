# SkinProAI: Explainable Multi-Model Dermatology Decision Support

## 1. System Architecture & AI Pipeline

SkinProAI is a clinician-first dermatology decision-support system that orchestrates multiple specialised AI models through a four-phase analysis pipeline. The system is designed to run entirely on-device using Google's MedGemma 4B — a 4-billion-parameter vision-language model purpose-built for edge deployment in clinical settings — executing on consumer hardware (Apple M4 Pro / MPS) in bfloat16 precision without requiring cloud GPU infrastructure.

### 1.1 Multi-Model Orchestration via MCP

Rather than relying on a single monolithic model, SkinProAI employs a Model Context Protocol (MCP) architecture that isolates each AI capability as an independent tool callable via JSON-RPC over subprocess stdio. This design provides fault isolation, independent model loading, and a clear audit trail of which model produced which output.

| Tool | Model | Purpose |
|------|-------|---------|
| `monet_analyze` | MONET (contrastive vision-language) | Extract 7 dermoscopic concept scores: ulceration, hair structures, vascular patterns, erythema, pigmented structures, gel artefacts, skin markings |
| `classify_lesion` | ConvNeXt Base + metadata MLP | 11-class lesion classification using dual-encoder architecture combining image features with MONET concept scores |
| `generate_gradcam` | Grad-CAM on ConvNeXt | Generate visual attention heatmap overlays highlighting regions driving the classification decision |
| `search_guidelines` | FAISS + all-MiniLM-L6-v2 | Retrieve relevant clinical guideline passages from a RAG index of 286 chunks across 7 dermatology PDFs |
| `compare_images` | MONET + overlay | Temporal change detection comparing dermoscopic feature vectors between sequential images |

All tools receive absolute file paths as input (never raw image data), enabling secure subprocess isolation between the main FastAPI process (Python 3.9) and the MCP server (Python 3.11).

### 1.2 Four-Phase Analysis Pipeline

Every image analysis follows a structured, transparent pipeline:

**Phase 1 — Independent Visual Examination.** MedGemma 4B performs a systematic dermoscopic assessment *before* seeing any AI tool output. It evaluates pattern architecture, colour distribution, border characteristics, and structural features, producing differential diagnoses ranked by clinical probability. This deliberate sequencing prevents anchoring bias — the language model forms its own clinical impression independently.

**Phase 2 — AI Classification Tools.** Three MCP tools execute in sequence: MONET extracts quantitative concept scores (0–1 range for each dermoscopic feature), ConvNeXt classifies the lesion into one of 11 diagnostic categories with calibrated probabilities, and Grad-CAM generates an attention overlay showing which image regions drove the classification. Each tool's output is streamed to the clinician in real-time with visual bar charts.

**Phase 3 — Reconciliation.** MedGemma receives both its own Phase 1 assessment and the Phase 2 tool outputs, then performs explicit agreement/disagreement analysis. It identifies where its visual findings align with or diverge from the quantitative classifiers, produces an integrated assessment with a stated confidence level, and explains its reasoning. This adversarial cross-check between independent assessments is central to the system's reliability.

**Phase 4 — Management Guidance with RAG.** The system automatically queries a FAISS-indexed knowledge base of clinical dermatology guidelines (BAD, NICE, and specialist PDFs covering BCC, SCC, melanoma, actinic keratosis, contact dermatitis, lichen sclerosus, and cutaneous warts). Using the diagnosed condition as a search query, the RAG system retrieves the top-5 most relevant guideline passages via cosine similarity over sentence-transformer embeddings (all-MiniLM-L6-v2, 384 dimensions). MedGemma then synthesises lesion-specific management recommendations — biopsy, excision, monitoring, or discharge — grounded in the retrieved evidence, with inline superscript citations linking back to source documents and page numbers.

---

## 2. AI Explainability

Explainability is not an afterthought in SkinProAI — it is embedded at every layer of the architecture. The system provides three complementary forms of explanation: visual, quantitative, and narrative.

### 2.1 Visual Explainability: Grad-CAM Attention Maps

Grad-CAM (Gradient-weighted Class Activation Mapping) hooks into the final convolutional layer of the ConvNeXt classifier to produce a spatial heatmap overlay on the original dermoscopic image. This directly answers the question "where is the model looking?" — showing clinicians which morphological features (border irregularity, colour variegation, structural asymmetry) are driving the classification. For temporal comparisons, the system generates side-by-side Grad-CAM pairs (previous vs. current) so clinicians can visually assess whether attention regions have shifted, expanded, or resolved.

### 2.2 Quantitative Explainability: MONET Concept Scores

MONET provides human-interpretable concept decomposition by scoring seven clinically meaningful dermoscopic features on a continuous 0–1 scale. Unlike black-box classifiers that output only a label and probability, MONET's concept scores reveal *why* a lesion received its classification: a high vascular score combined with low pigmentation explains a vascular lesion diagnosis; high pigmented-structure scores with border irregularity support melanocytic concern. These scores are rendered as visual bar charts in the streaming UI, giving clinicians immediate quantitative insight into the feature profile driving the AI's assessment.

When comparing sequential images over time, the system computes MONET feature deltas — the signed difference in each concept score between timepoints — enabling objective quantification of lesion evolution. A change from 0.3 to 0.7 in the vascular score, for example, signals new vessel formation that warrants clinical attention, independent of any subjective visual comparison.

### 2.3 Narrative Explainability: MedGemma Reasoning Transparency

The streaming interface exposes MedGemma's reasoning process through structured markup segments. `[THINKING]` blocks display the model's intermediate reasoning (differential construction, feature weighting, agreement analysis) in real-time, with animated spinners that resolve to completion indicators as each reasoning phase finishes. `[RESPONSE]` blocks contain the synthesised clinical narrative. This staged transparency allows clinicians to follow the AI's analytical process rather than receiving only a final pronouncement.

The reconciliation phase (Phase 3) is particularly significant for explainability: MedGemma explicitly states where its independent visual assessment agrees or disagrees with the quantitative classifiers, and explains the basis for its integrated conclusion. This adversarial structure makes disagreements visible and auditable.

### 2.4 Evidence Grounding: RAG Citations

Management recommendations in Phase 4 include inline superscript references (e.g., "Wide local excision with 2cm margins is recommended for tumours >2mm Breslow thickness¹") linked to specific guideline documents and page numbers. This evidence chain — from diagnosis through to management recommendation — is fully traceable to published clinical guidelines, supporting regulatory compliance and clinical governance requirements.

---

## 3. Clinician-First Design & User Interface

### 3.1 Design Philosophy

SkinProAI is built around the principle that AI should augment clinical decision-making, not replace it. The interface is designed for the workflow of a clinician reviewing dermoscopic images: patient selection, lesion documentation, temporal tracking, and structured analysis with clear next-step guidance.

The system deliberately avoids presenting AI conclusions as definitive diagnoses. Instead, results are framed as ranked differentials with calibrated confidence scores, supported by quantitative feature evidence and visual attention maps. The clinician retains full agency over the diagnostic and management decision.

### 3.2 Streaming Real-Time Interface

The UI employs Server-Sent Events (SSE) to stream analysis output in real-time, maintaining clinician engagement during model inference. Rather than a loading spinner followed by a wall of text, clinicians observe the analysis unfolding phase by phase:

- **Tool status lines** appear as compact single-line indicators with animated spinners that resolve to green completion dots, showing tool name and summary result (e.g., "ConvNeXt classification — Melanoma (89%)")
- **Thinking indicators** display MedGemma's reasoning steps with spinners that transition to done states as each phase completes
- **Streamed text** appears word-by-word, allowing clinicians to begin reading findings before analysis completes

This streaming pattern reduces perceived latency and provides transparency into the multi-model pipeline's progress.

### 3.3 Temporal Lesion Tracking

The data model supports longitudinal monitoring through a Patient → Lesion → LesionImage hierarchy. Each lesion maintains a timeline of images with timestamps, and the system automatically triggers temporal comparison when a new image is uploaded for a previously-analysed lesion. Comparison results include MONET feature deltas, side-by-side Grad-CAM overlays, and a status classification (Stable / Minor Change / Significant Change / Improved) rendered with colour-coded indicators.

This temporal architecture supports the clinical workflow for monitoring suspicious lesions over time — a common scenario in dermatology where serial dermoscopy is preferred over immediate biopsy for lesions with intermediate risk profiles.

### 3.4 Conversational Follow-Up

After analysis completes, the system transitions to a conversational interface where clinicians can ask follow-up questions grounded in the analysis context. The chat maintains full awareness of the diagnosed condition, MONET features, classification results, and guideline context, enabling queries like "what are the excision margin recommendations for this depth?" or "how does the asymmetry score compare to the previous image?" The text-only chat pathway routes through MedGemma with the full analysis state, ensuring responses remain clinically contextualised.

### 3.5 Edge Deployment Architecture

SkinProAI runs entirely on-device using MedGemma 4B in bfloat16 precision on consumer hardware (demonstrated on Apple M4 Pro with 24GB RAM using Metal Performance Shaders). No patient data leaves the device — all inference, image storage, and guideline retrieval execute locally. This edge-first architecture addresses data sovereignty requirements in clinical settings where patient images cannot be transmitted to cloud services. The system also supports containerised deployment via Docker for institutional environments, with optional GPU acceleration on CUDA-equipped hardware or HuggingFace Spaces.

### 3.6 Interface Components

The frontend is built in React 18 with a glassmorphism-inspired design language. Key interface elements include:

- **Patient grid** — Card-based patient selection with creation workflow
- **Chat interface** — Unified conversation view combining image analysis, tool outputs, and text chat in a single scrollable thread
- **Tool call cards** — Inline status indicators for each AI tool invocation (analyse, classify, Grad-CAM, guidelines search, compare) with expandable result summaries
- **Image upload** — Drag-and-drop or click-to-upload with preview, supporting both initial analysis and temporal comparison workflows
- **Post-analysis prompt** — Contextual hint guiding clinicians to ask follow-up questions, provide additional context, or upload comparison images
- **Markdown rendering** — Clinical narratives rendered with proper formatting, headers, lists, and inline citations for readability

The interface prioritises information density without clutter: tool outputs collapse to single-line summaries by default, reasoning phases are visually distinguished from conclusions, and temporal comparisons use colour-coded status indicators (green/amber/red/blue) that map to clinical urgency levels.
