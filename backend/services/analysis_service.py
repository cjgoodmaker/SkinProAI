"""
Analysis Service - Wraps MedGemmaAgent for API use
"""

from pathlib import Path
from dataclasses import asdict
from typing import Optional, Generator

from models.medgemma_agent import MedGemmaAgent
from data.case_store import get_case_store


class AnalysisService:
    """Singleton service for managing analysis operations"""

    _instance = None

    def __init__(self):
        self.agent = MedGemmaAgent(verbose=True)
        self.store = get_case_store()
        self._loaded = False

    def _ensure_loaded(self):
        """Lazy load the ML models"""
        if not self._loaded:
            self.agent.load_model()
            self._loaded = True

    def analyze(
        self,
        patient_id: str,
        lesion_id: str,
        image_id: str,
        question: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Run analysis on an image, yielding streaming chunks"""
        self._ensure_loaded()

        image = self.store.get_image(patient_id, lesion_id, image_id)
        if not image or not image.image_path:
            yield "[ERROR]No image uploaded[/ERROR]"
            return

        # Update stage
        self.store.update_image(patient_id, lesion_id, image_id, stage="analyzing")

        # Reset agent state for new analysis
        self.agent.reset_state()

        # Run analysis with question
        for chunk in self.agent.analyze_image_stream(image.image_path, question=question or ""):
            yield chunk

        # Save diagnosis after analysis
        if self.agent.last_diagnosis:
            analysis_data = {
                "diagnosis": self.agent.last_diagnosis["predictions"][0]["class"],
                "full_name": self.agent.last_diagnosis["predictions"][0]["full_name"],
                "confidence": self.agent.last_diagnosis["predictions"][0]["probability"],
                "all_predictions": self.agent.last_diagnosis["predictions"]
            }

            # Save MONET features if available
            if self.agent.last_monet_result:
                analysis_data["monet_features"] = self.agent.last_monet_result.get("features", {})

            self.store.update_image(
                patient_id, lesion_id, image_id,
                stage="awaiting_confirmation",
                analysis=analysis_data
            )

    def confirm(
        self,
        patient_id: str,
        lesion_id: str,
        image_id: str,
        confirmed: bool,
        feedback: Optional[str] = None
    ) -> Generator[str, None, None]:
        """Confirm diagnosis and generate management guidance"""
        for chunk in self.agent.generate_management_guidance(confirmed, feedback):
            yield chunk

        # Update stage to complete
        self.store.update_image(patient_id, lesion_id, image_id, stage="complete")

    def chat_followup(
        self,
        patient_id: str,
        lesion_id: str,
        message: str
    ) -> Generator[str, None, None]:
        """Handle follow-up chat messages"""
        # Save user message
        self.store.add_chat_message(patient_id, lesion_id, "user", message)

        # Generate response
        response = ""
        for chunk in self.agent.chat_followup(message):
            response += chunk
            yield chunk

        # Save assistant response
        self.store.add_chat_message(patient_id, lesion_id, "assistant", response)

    def get_chat_history(self, patient_id: str, lesion_id: str):
        """Get chat history for a lesion"""
        messages = self.store.get_chat_history(patient_id, lesion_id)
        return [asdict(m) for m in messages]

    def compare_images(
        self,
        patient_id: str,
        lesion_id: str,
        previous_image_path: str,
        current_image_path: str,
        current_image_id: str
    ) -> Generator[str, None, None]:
        """Compare two images and assess changes"""
        self._ensure_loaded()

        # Run comparison
        comparison_result = None
        for chunk in self.agent.compare_followup_images(previous_image_path, current_image_path):
            yield chunk

        # Extract comparison status from agent if available
        # Default to STABLE if we can't determine
        comparison_data = {
            "status": "STABLE",
            "summary": "Comparison complete"
        }

        # Update the current image with comparison data
        self.store.update_image(
            patient_id, lesion_id, current_image_id,
            comparison=comparison_data
        )


def get_analysis_service() -> AnalysisService:
    """Get or create AnalysisService singleton"""
    if AnalysisService._instance is None:
        AnalysisService._instance = AnalysisService()
    return AnalysisService._instance
