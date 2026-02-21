"""
Chat Service - Patient-level chat with tool dispatch and streaming
"""

import io
import re
import uuid
from typing import Generator, Optional
from pathlib import Path
from PIL import Image as PILImage

from data.case_store import get_case_store
from backend.services.analysis_service import get_analysis_service


def _extract_response_text(raw: str) -> str:
    """Pull clean text out of [RESPONSE]...[/RESPONSE]; strip all other tags."""
    # Grab the RESPONSE block first
    match = re.search(r'\[RESPONSE\](.*?)\[/RESPONSE\]', raw, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: strip every known markup tag
    clean = re.sub(
        r'\[(STAGE:[^\]]+|THINKING|RESPONSE|/RESPONSE|/THINKING|/STAGE'
        r'|ERROR|/ERROR|RESULT|/RESULT|CONFIRM:\d+|/CONFIRM)\]',
        '', raw
    )
    return clean.strip()


class ChatService:
    _instance = None

    def __init__(self):
        self.store = get_case_store()

    def _get_image_url(self, patient_id: str, lesion_id: str, image_id: str) -> str:
        return f"/uploads/{patient_id}/{lesion_id}/{image_id}/image.png"

    def stream_chat(
        self,
        patient_id: str,
        content: str,
        image_bytes: Optional[bytes] = None,
    ) -> Generator[dict, None, None]:
        """Main chat handler — yields SSE event dicts."""
        analysis_service = get_analysis_service()

        if image_bytes:
            # ----------------------------------------------------------------
            # Image path: analyze (and optionally compare).
            # We do NOT stream the raw verbose analysis text to the chat bubble —
            # the tool card IS the display artefact.  We accumulate the text
            # internally, extract the clean [RESPONSE] block, and put it in
            # tool_result.summary so the expanded card can show it.
            # ----------------------------------------------------------------
            lesion = self.store.get_or_create_chat_lesion(patient_id)

            img_record = self.store.add_image(patient_id, lesion.id)
            pil_image = PILImage.open(io.BytesIO(image_bytes)).convert("RGB")
            abs_path = self.store.save_lesion_image(
                patient_id, lesion.id, img_record.id, pil_image
            )
            self.store.update_image(patient_id, lesion.id, img_record.id, image_path=abs_path)

            user_image_url = self._get_image_url(patient_id, lesion.id, img_record.id)
            self.store.add_patient_chat_message(
                patient_id, "user", content, image_url=user_image_url
            )

            # ---- tool: analyze_image ----------------------------------------
            call_id = f"tc-{uuid.uuid4().hex[:6]}"
            yield {"type": "tool_start", "tool": "analyze_image", "call_id": call_id}

            analysis_text = ""
            for chunk in analysis_service.analyze(patient_id, lesion.id, img_record.id):
                yield {"type": "text", "content": chunk}
                analysis_text += chunk

            updated_img = self.store.get_image(patient_id, lesion.id, img_record.id)
            analysis_result: dict = {
                "image_url": user_image_url,
                "summary": _extract_response_text(analysis_text),
                "diagnosis": None,
                "full_name": None,
                "confidence": None,
                "all_predictions": [],
            }
            if updated_img and updated_img.analysis:
                a = updated_img.analysis
                analysis_result.update({
                    "diagnosis": a.get("diagnosis"),
                    "full_name": a.get("full_name"),
                    "confidence": a.get("confidence"),
                    "all_predictions": a.get("all_predictions", []),
                })

            yield {
                "type": "tool_result",
                "tool": "analyze_image",
                "call_id": call_id,
                "result": analysis_result,
            }

            # ---- tool: compare_images (if a previous image exists) ----------
            previous_img = self.store.get_previous_image(patient_id, lesion.id, img_record.id)
            compare_call_id = None
            compare_result = None
            compare_text = ""

            if (
                previous_img
                and previous_img.image_path
                and Path(previous_img.image_path).exists()
            ):
                compare_call_id = f"tc-{uuid.uuid4().hex[:6]}"
                yield {
                    "type": "tool_start",
                    "tool": "compare_images",
                    "call_id": compare_call_id,
                }

                for chunk in analysis_service.compare_images(
                    patient_id,
                    lesion.id,
                    previous_img.image_path,
                    abs_path,
                    img_record.id,
                ):
                    yield {"type": "text", "content": chunk}
                    compare_text += chunk

                updated_img2 = self.store.get_image(patient_id, lesion.id, img_record.id)
                compare_result = {
                    "prev_image_url": self._get_image_url(patient_id, lesion.id, previous_img.id),
                    "curr_image_url": user_image_url,
                    "status_label": "STABLE",
                    "feature_changes": {},
                    "summary": _extract_response_text(compare_text),
                }
                if updated_img2 and updated_img2.comparison:
                    c = updated_img2.comparison
                    compare_result.update({
                        "status_label": c.get("status", "STABLE"),
                        "feature_changes": c.get("feature_changes", {}),
                    })
                    if c.get("summary"):
                        compare_result["summary"] = c["summary"]

                yield {
                    "type": "tool_result",
                    "tool": "compare_images",
                    "call_id": compare_call_id,
                    "result": compare_result,
                }

            # Save assistant message
            tool_calls_data = [{
                "id": call_id,
                "tool": "analyze_image",
                "status": "complete",
                "result": analysis_result,
            }]
            if compare_call_id and compare_result:
                tool_calls_data.append({
                    "id": compare_call_id,
                    "tool": "compare_images",
                    "status": "complete",
                    "result": compare_result,
                })

            self.store.add_patient_chat_message(
                patient_id, "assistant", analysis_text + compare_text,
                tool_calls=tool_calls_data,
            )

        else:
            # ----------------------------------------------------------------
            # Text-only chat — stream chunks; tags are stripped on the frontend
            # ----------------------------------------------------------------
            self.store.add_patient_chat_message(patient_id, "user", content)

            analysis_service._ensure_loaded()
            response_text = ""
            for chunk in analysis_service.agent.chat_followup(content):
                yield {"type": "text", "content": chunk}
                response_text += chunk

            self.store.add_patient_chat_message(
                patient_id, "assistant", _extract_response_text(response_text)
            )


def get_chat_service() -> ChatService:
    if ChatService._instance is None:
        ChatService._instance = ChatService()
    return ChatService._instance
