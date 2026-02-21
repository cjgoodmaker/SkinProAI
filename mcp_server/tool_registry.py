"""
Lazy singleton loader for all 4 ML models used by the MCP server.
Fixes sys.path so the subprocess can import from models/.
"""

import sys
import os

# Ensure project root is on path (this file lives at project_root/mcp_server/tool_registry.py)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_monet = None
_convnext = None
_gradcam = None
_rag = None


def get_monet():
    global _monet
    if _monet is None:
        from models.monet_tool import MonetTool
        _monet = MonetTool()
        _monet.load()
    return _monet


def get_convnext():
    global _convnext
    if _convnext is None:
        from models.convnext_classifier import ConvNeXtClassifier
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _convnext = ConvNeXtClassifier(
            checkpoint_path=os.path.join(root, "models", "seed42_fold0.pt")
        )
        _convnext.load()
    return _convnext


def get_gradcam():
    global _gradcam
    if _gradcam is None:
        from models.gradcam_tool import GradCAMTool
        _gradcam = GradCAMTool(classifier=get_convnext())
        _gradcam.load()
    return _gradcam


def get_rag():
    global _rag
    if _rag is None:
        from models.guidelines_rag import get_guidelines_rag
        _rag = get_guidelines_rag()
        if not _rag.loaded:
            _rag.load_index()
    return _rag
