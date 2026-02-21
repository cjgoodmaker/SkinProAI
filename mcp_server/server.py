"""
SkinProAI MCP Server - Pure JSON-RPC 2.0 stdio server (no mcp library required).

Uses sys.executable (venv Python) so all ML packages (torch, transformers, etc.)
are available. Tools are loaded lazily on first call.

Run standalone: python mcp_server/server.py
(Should start silently, waiting on stdin.)
"""

import sys
import json
import os

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcp_server.tool_registry import get_monet, get_convnext, get_gradcam, get_rag


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _monet_analyze(arguments: dict) -> dict:
    from PIL import Image
    image = Image.open(arguments["image_path"]).convert("RGB")
    return get_monet().analyze(image)


def _classify_lesion(arguments: dict) -> dict:
    from PIL import Image
    image = Image.open(arguments["image_path"]).convert("RGB")
    monet_scores = arguments.get("monet_scores")
    return get_convnext().classify(
        clinical_image=image,
        derm_image=None,
        monet_scores=monet_scores,
    )


def _generate_gradcam(arguments: dict) -> dict:
    from PIL import Image
    import tempfile
    image = Image.open(arguments["image_path"]).convert("RGB")
    result = get_gradcam().analyze(image)

    gradcam_file = tempfile.NamedTemporaryFile(suffix="_gradcam.png", delete=False)
    gradcam_path = gradcam_file.name
    gradcam_file.close()
    result["overlay"].save(gradcam_path)

    return {
        "gradcam_path": gradcam_path,
        "predicted_class": result["predicted_class"],
        "predicted_class_full": result["predicted_class_full"],
        "confidence": result["confidence"],
    }


def _search_guidelines(arguments: dict) -> dict:
    query = arguments.get("query", "")
    diagnosis = arguments.get("diagnosis") or ""
    rag = get_rag()
    context, references = rag.get_management_context(diagnosis, query)
    references_display = rag.format_references_for_display(references)
    return {
        "context": context,
        "references": references,
        "references_display": references_display,
    }


def _compare_images(arguments: dict) -> dict:
    from PIL import Image
    import tempfile
    image1 = Image.open(arguments["image1_path"]).convert("RGB")
    image2 = Image.open(arguments["image2_path"]).convert("RGB")

    from models.overlay_tool import get_overlay_tool
    comparison = get_overlay_tool().generate_comparison_overlay(
        image1, image2, label1="Previous", label2="Current"
    )
    comparison_path = comparison["path"]

    monet = get_monet()
    prev_result = monet.analyze(image1)
    curr_result = monet.analyze(image2)

    monet_deltas = {}
    for name in curr_result["features"]:
        prev_val = prev_result["features"].get(name, 0.0)
        curr_val = curr_result["features"][name]
        delta = curr_val - prev_val
        if abs(delta) > 0.1:
            monet_deltas[name] = {
                "previous": prev_val,
                "current": curr_val,
                "delta": delta,
            }

    # Generate GradCAM for both images so the frontend can show a side-by-side comparison
    prev_gradcam_path = None
    curr_gradcam_path = None
    try:
        gradcam = get_gradcam()
        prev_gc = gradcam.analyze(image1)
        curr_gc = gradcam.analyze(image2)

        f1 = tempfile.NamedTemporaryFile(suffix="_gradcam.png", delete=False)
        prev_gradcam_path = f1.name
        f1.close()
        prev_gc["overlay"].save(prev_gradcam_path)

        f2 = tempfile.NamedTemporaryFile(suffix="_gradcam.png", delete=False)
        curr_gradcam_path = f2.name
        f2.close()
        curr_gc["overlay"].save(curr_gradcam_path)
    except Exception:
        pass  # GradCAM comparison is best-effort

    return {
        "comparison_path": comparison_path,
        "monet_deltas": monet_deltas,
        "prev_gradcam_path": prev_gradcam_path,
        "curr_gradcam_path": curr_gradcam_path,
    }


TOOLS = {
    "monet_analyze": _monet_analyze,
    "classify_lesion": _classify_lesion,
    "generate_gradcam": _generate_gradcam,
    "search_guidelines": _search_guidelines,
    "compare_images": _compare_images,
}

TOOLS_LIST = [
    {
        "name": "monet_analyze",
        "description": "Extract MONET concept-presence scores from a skin lesion image.",
        "inputSchema": {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
            "required": ["image_path"],
        },
    },
    {
        "name": "classify_lesion",
        "description": "Classify a skin lesion using ConvNeXt dual-encoder.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image_path": {"type": "string"},
                "monet_scores": {"type": "array"},
            },
            "required": ["image_path"],
        },
    },
    {
        "name": "generate_gradcam",
        "description": "Generate a Grad-CAM attention overlay for a skin lesion image.",
        "inputSchema": {
            "type": "object",
            "properties": {"image_path": {"type": "string"}},
            "required": ["image_path"],
        },
    },
    {
        "name": "search_guidelines",
        "description": "Search clinical guidelines RAG for management context.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "diagnosis": {"type": "string"},
            },
            "required": ["query"],
        },
    },
    {
        "name": "compare_images",
        "description": "Generate comparison overlay and MONET deltas for two lesion images.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "image1_path": {"type": "string"},
                "image2_path": {"type": "string"},
            },
            "required": ["image1_path", "image2_path"],
        },
    },
]


# ---------------------------------------------------------------------------
# JSON-RPC 2.0 dispatcher
# ---------------------------------------------------------------------------

def handle_request(request: dict):
    method = request.get("method")
    req_id = request.get("id")  # None for notifications
    params = request.get("params", {})

    if method == "initialize":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {"listChanged": False}},
                "serverInfo": {"name": "SkinProAI", "version": "1.0.0"},
            },
        }

    if method in ("notifications/initialized",):
        return None  # notification — no response

    if method == "tools/list":
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "result": {"tools": TOOLS_LIST},
        }

    if method == "tools/call":
        name = params.get("name")
        arguments = params.get("arguments", {})
        if name not in TOOLS:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Unknown tool: {name}"},
            }
        try:
            result = TOOLS[name](arguments)
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": json.dumps(result)}],
                    "isError": False,
                },
            }
        except Exception as e:
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [{"type": "text", "text": f"Tool error: {e}"}],
                    "isError": True,
                },
            }

    # Unknown method with id → method not found
    if req_id is not None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    return None  # unknown notification — ignore


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = handle_request(request)
        if response is not None:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()


if __name__ == "__main__":
    main()
