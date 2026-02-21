"""
Case Store - JSON-based persistence for patients, lesions, and images
"""

import json
import uuid
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field, asdict
from PIL import Image as PILImage


@dataclass
class ChatMessage:
    role: str  # "user" or "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class LesionImage:
    """A single image capture of a lesion at a point in time"""
    id: str
    lesion_id: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    image_path: Optional[str] = None
    gradcam_path: Optional[str] = None
    analysis: Optional[Dict[str, Any]] = None  # {diagnosis, confidence, monet_features}
    comparison: Optional[Dict[str, Any]] = None  # {status, feature_changes, summary}
    is_original: bool = False
    stage: str = "pending"  # pending, analyzing, complete, error


@dataclass
class Lesion:
    """A tracked lesion that can have multiple images over time"""
    id: str
    patient_id: str
    name: str  # User-provided label (e.g., "Left shoulder mole")
    location: str = ""  # Body location
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    chat_history: List[Dict] = field(default_factory=list)


@dataclass
class Patient:
    """A patient who can have multiple lesions"""
    id: str
    name: str
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class CaseStore:
    """JSON-based persistence for patients, lesions, and images"""

    def __init__(self, data_dir: str = None):
        if data_dir is None:
            data_dir = Path(__file__).parent
        self.data_dir = Path(data_dir)
        self.patients_file = self.data_dir / "patients.json"
        self.lesions_dir = self.data_dir / "lesions"
        self.uploads_dir = self.data_dir / "uploads"

        # Ensure directories exist
        self.lesions_dir.mkdir(parents=True, exist_ok=True)
        self.uploads_dir.mkdir(parents=True, exist_ok=True)

        # Initialize patients file if needed
        if not self.patients_file.exists():
            self._init_patients_file()

    def _init_patients_file(self):
        """Initialize patients file"""
        data = {"patients": []}
        with open(self.patients_file, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_patients_data(self) -> Dict:
        """Load patients JSON file"""
        with open(self.patients_file, 'r') as f:
            return json.load(f)

    def _save_patients_data(self, data: Dict):
        """Save patients JSON file"""
        with open(self.patients_file, 'w') as f:
            json.dump(data, f, indent=2)

    # -------------------------------------------------------------------------
    # Patient Methods
    # -------------------------------------------------------------------------

    def list_patients(self) -> List[Patient]:
        """List all patients"""
        data = self._load_patients_data()
        return [Patient(**p) for p in data.get("patients", [])]

    def get_patient(self, patient_id: str) -> Optional[Patient]:
        """Get a patient by ID"""
        data = self._load_patients_data()
        for p in data.get("patients", []):
            if p["id"] == patient_id:
                return Patient(**p)
        return None

    def create_patient(self, name: str) -> Patient:
        """Create a new patient"""
        patient = Patient(
            id=f"patient-{uuid.uuid4().hex[:8]}",
            name=name
        )

        data = self._load_patients_data()
        data["patients"].append(asdict(patient))
        self._save_patients_data(data)

        # Create directory for this patient's lesions
        (self.lesions_dir / patient.id).mkdir(exist_ok=True)

        return patient

    def delete_patient(self, patient_id: str):
        """Delete a patient and all their lesions"""
        data = self._load_patients_data()
        data["patients"] = [p for p in data["patients"] if p["id"] != patient_id]
        self._save_patients_data(data)

        # Delete lesion files
        patient_lesions_dir = self.lesions_dir / patient_id
        if patient_lesions_dir.exists():
            shutil.rmtree(patient_lesions_dir)

        # Delete uploads
        patient_uploads_dir = self.uploads_dir / patient_id
        if patient_uploads_dir.exists():
            shutil.rmtree(patient_uploads_dir)

        # Delete patient chat history
        patient_chat_file = self.data_dir / "patient_chats" / f"{patient_id}.json"
        if patient_chat_file.exists():
            patient_chat_file.unlink()

    def get_patient_lesion_count(self, patient_id: str) -> int:
        """Get number of lesions for a patient"""
        return len(self.list_lesions(patient_id))

    # -------------------------------------------------------------------------
    # Lesion Methods
    # -------------------------------------------------------------------------

    def _get_lesion_path(self, patient_id: str, lesion_id: str) -> Path:
        """Get path to lesion JSON file"""
        return self.lesions_dir / patient_id / f"{lesion_id}.json"

    def list_lesions(self, patient_id: str) -> List[Lesion]:
        """List all lesions for a patient"""
        patient_dir = self.lesions_dir / patient_id
        if not patient_dir.exists():
            return []

        lesions = []
        for f in sorted(patient_dir.glob("*.json")):
            with open(f, 'r') as fp:
                data = json.load(fp)
                # Only load lesion data, not images
                lesion_data = {k: v for k, v in data.items() if k != 'images'}
                lesions.append(Lesion(**lesion_data))

        lesions.sort(key=lambda x: x.created_at)
        return lesions

    def get_lesion(self, patient_id: str, lesion_id: str) -> Optional[Lesion]:
        """Get a lesion by ID"""
        path = self._get_lesion_path(patient_id, lesion_id)
        if not path.exists():
            return None

        with open(path, 'r') as f:
            data = json.load(f)
            lesion_data = {k: v for k, v in data.items() if k != 'images'}
            return Lesion(**lesion_data)

    def create_lesion(self, patient_id: str, name: str, location: str = "") -> Lesion:
        """Create a new lesion for a patient"""
        lesion = Lesion(
            id=f"lesion-{uuid.uuid4().hex[:8]}",
            patient_id=patient_id,
            name=name,
            location=location
        )

        # Ensure patient directory exists
        patient_dir = self.lesions_dir / patient_id
        patient_dir.mkdir(exist_ok=True)

        # Save lesion with empty images array
        self._save_lesion_data(patient_id, lesion.id, {
            **asdict(lesion),
            "images": []
        })

        return lesion

    def _save_lesion_data(self, patient_id: str, lesion_id: str, data: Dict):
        """Save lesion data to JSON file"""
        path = self._get_lesion_path(patient_id, lesion_id)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def _load_lesion_data(self, patient_id: str, lesion_id: str) -> Optional[Dict]:
        """Load full lesion data including images"""
        path = self._get_lesion_path(patient_id, lesion_id)
        if not path.exists():
            return None

        with open(path, 'r') as f:
            return json.load(f)

    def delete_lesion(self, patient_id: str, lesion_id: str):
        """Delete a lesion and all its images"""
        path = self._get_lesion_path(patient_id, lesion_id)
        if path.exists():
            path.unlink()

        # Delete uploads for this lesion
        lesion_uploads_dir = self.uploads_dir / patient_id / lesion_id
        if lesion_uploads_dir.exists():
            shutil.rmtree(lesion_uploads_dir)

    def update_lesion(self, patient_id: str, lesion_id: str, name: str = None, location: str = None):
        """Update lesion name or location"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            return

        if name is not None:
            data["name"] = name
        if location is not None:
            data["location"] = location

        self._save_lesion_data(patient_id, lesion_id, data)

    # -------------------------------------------------------------------------
    # LesionImage Methods
    # -------------------------------------------------------------------------

    def list_images(self, patient_id: str, lesion_id: str) -> List[LesionImage]:
        """List all images for a lesion"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            return []

        images = [LesionImage(**img) for img in data.get("images", [])]
        images.sort(key=lambda x: x.timestamp)
        return images

    def get_image(self, patient_id: str, lesion_id: str, image_id: str) -> Optional[LesionImage]:
        """Get an image by ID"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            return None

        for img in data.get("images", []):
            if img["id"] == image_id:
                return LesionImage(**img)
        return None

    def add_image(self, patient_id: str, lesion_id: str) -> LesionImage:
        """Add a new image to a lesion's timeline"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            raise ValueError(f"Lesion {lesion_id} not found")

        # Check if this is the first image
        is_first = len(data.get("images", [])) == 0

        image = LesionImage(
            id=f"img-{uuid.uuid4().hex[:8]}",
            lesion_id=lesion_id,
            is_original=is_first
        )

        if "images" not in data:
            data["images"] = []
        data["images"].append(asdict(image))
        self._save_lesion_data(patient_id, lesion_id, data)

        return image

    def update_image(
        self,
        patient_id: str,
        lesion_id: str,
        image_id: str,
        image_path: str = None,
        gradcam_path: str = None,
        analysis: Dict = None,
        comparison: Dict = None,
        stage: str = None
    ):
        """Update an image's data"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            return

        for img in data.get("images", []):
            if img["id"] == image_id:
                if image_path is not None:
                    img["image_path"] = image_path
                if gradcam_path is not None:
                    img["gradcam_path"] = gradcam_path
                if analysis is not None:
                    img["analysis"] = analysis
                if comparison is not None:
                    img["comparison"] = comparison
                if stage is not None:
                    img["stage"] = stage
                break

        self._save_lesion_data(patient_id, lesion_id, data)

    def save_lesion_image(
        self,
        patient_id: str,
        lesion_id: str,
        image_id: str,
        image: PILImage.Image,
        filename: str = "image.png"
    ) -> str:
        """Save an uploaded image file, return the path"""
        upload_dir = self.uploads_dir / patient_id / lesion_id / image_id
        upload_dir.mkdir(parents=True, exist_ok=True)

        image_path = upload_dir / filename
        image.save(image_path)

        return str(image_path)

    def get_previous_image(
        self,
        patient_id: str,
        lesion_id: str,
        current_image_id: str
    ) -> Optional[LesionImage]:
        """Get the image before the current one (for comparison)"""
        images = self.list_images(patient_id, lesion_id)

        for i, img in enumerate(images):
            if img.id == current_image_id and i > 0:
                return images[i - 1]
        return None

    # -------------------------------------------------------------------------
    # Chat Methods (scoped to lesion)
    # -------------------------------------------------------------------------

    def add_chat_message(self, patient_id: str, lesion_id: str, role: str, content: str):
        """Add a chat message to a lesion"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            return

        message = ChatMessage(role=role, content=content)
        if "chat_history" not in data:
            data["chat_history"] = []
        data["chat_history"].append(asdict(message))
        self._save_lesion_data(patient_id, lesion_id, data)

    def get_chat_history(self, patient_id: str, lesion_id: str) -> List[ChatMessage]:
        """Get chat history for a lesion"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            return []

        return [ChatMessage(**m) for m in data.get("chat_history", [])]

    def clear_chat_history(self, patient_id: str, lesion_id: str):
        """Clear chat history for a lesion"""
        data = self._load_lesion_data(patient_id, lesion_id)
        if data is None:
            return

        data["chat_history"] = []
        self._save_lesion_data(patient_id, lesion_id, data)

    # -------------------------------------------------------------------------
    # Patient-level Chat Methods
    # -------------------------------------------------------------------------

    def _get_patient_chat_file(self, patient_id: str) -> Path:
        """Get path to patient-level chat JSON file"""
        chat_dir = self.data_dir / "patient_chats"
        chat_dir.mkdir(exist_ok=True)
        return chat_dir / f"{patient_id}.json"

    def get_patient_chat_history(self, patient_id: str) -> List[dict]:
        """Get chat history for a patient"""
        chat_file = self._get_patient_chat_file(patient_id)
        if not chat_file.exists():
            return []
        with open(chat_file, 'r') as f:
            data = json.load(f)
        return data.get("messages", [])

    def add_patient_chat_message(
        self,
        patient_id: str,
        role: str,
        content: str,
        image_url: Optional[str] = None,
        tool_calls: Optional[list] = None
    ):
        """Add a message to patient-level chat history"""
        chat_file = self._get_patient_chat_file(patient_id)
        if chat_file.exists():
            with open(chat_file, 'r') as f:
                data = json.load(f)
        else:
            data = {"messages": []}

        message: Dict[str, Any] = {
            "id": f"msg-{uuid.uuid4().hex[:8]}",
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
        }
        if image_url is not None:
            message["image_url"] = image_url
        if tool_calls is not None:
            message["tool_calls"] = tool_calls

        data["messages"].append(message)
        with open(chat_file, 'w') as f:
            json.dump(data, f, indent=2)

    def clear_patient_chat_history(self, patient_id: str):
        """Clear patient-level chat history"""
        chat_file = self._get_patient_chat_file(patient_id)
        with open(chat_file, 'w') as f:
            json.dump({"messages": []}, f)

    def get_or_create_chat_lesion(self, patient_id: str) -> 'Lesion':
        """Get or create the internal chat-images lesion for a patient"""
        for lesion in self.list_lesions(patient_id):
            if lesion.name == "__chat_images__":
                return lesion
        return self.create_lesion(patient_id, "__chat_images__", "internal")

    def get_latest_chat_image(self, patient_id: str) -> Optional['LesionImage']:
        """Get the most recently analyzed chat image for a patient"""
        lesion = self.get_or_create_chat_lesion(patient_id)
        images = self.list_images(patient_id, lesion.id)
        for img in reversed(images):
            if img.analysis is not None:
                return img
        return None


# Singleton instance
_store_instance = None


def get_case_store() -> CaseStore:
    """Get or create CaseStore singleton"""
    global _store_instance
    if _store_instance is None:
        _store_instance = CaseStore()
    return _store_instance


if __name__ == "__main__":
    # Test the store
    store = CaseStore()

    print("Patients:")
    for patient in store.list_patients():
        print(f"  - {patient.id}: {patient.name}")

    # Create a test patient
    print("\nCreating test patient...")
    patient = store.create_patient("Test Patient")
    print(f"  Created: {patient.id}")

    # Create a lesion
    print("\nCreating lesion...")
    lesion = store.create_lesion(patient.id, "Left shoulder mole", "Left shoulder")
    print(f"  Created: {lesion.id}")

    # Add an image
    print("\nAdding image...")
    image = store.add_image(patient.id, lesion.id)
    print(f"  Created: {image.id} (is_original={image.is_original})")

    # Add another image
    image2 = store.add_image(patient.id, lesion.id)
    print(f"  Created: {image2.id} (is_original={image2.is_original})")

    # List images
    print(f"\nImages for lesion {lesion.id}:")
    for img in store.list_images(patient.id, lesion.id):
        print(f"  - {img.id}: original={img.is_original}, stage={img.stage}")

    # Cleanup
    print("\nCleaning up test patient...")
    store.delete_patient(patient.id)
    print("Done!")
