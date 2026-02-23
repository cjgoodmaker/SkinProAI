"""
ConvNeXt Classifier Tool - Skin lesion classification using ConvNeXt + MONET features
Loads seed42_fold0.pt checkpoint and performs classification.
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, Dict, List, Tuple
import timm


# Class names for the 11-class skin lesion classification
CLASS_NAMES = [
    'AKIEC', 'BCC', 'BEN_OTH', 'BKL', 'DF',
    'INF', 'MAL_OTH', 'MEL', 'NV', 'SCCKA', 'VASC'
]

CLASS_FULL_NAMES = {
    'AKIEC': 'Actinic Keratosis / Intraepithelial Carcinoma',
    'BCC': 'Basal Cell Carcinoma',
    'BEN_OTH': 'Benign Other',
    'BKL': 'Benign Keratosis-like Lesion',
    'DF': 'Dermatofibroma',
    'INF': 'Inflammatory',
    'MAL_OTH': 'Malignant Other',
    'MEL': 'Melanoma',
    'NV': 'Melanocytic Nevus',
    'SCCKA': 'Squamous Cell Carcinoma / Keratoacanthoma',
    'VASC': 'Vascular Lesion'
}


class ConvNeXtDualEncoder(nn.Module):
    """
    Dual-image ConvNeXt model matching the trained checkpoint.
    Processes BOTH clinical and dermoscopy images through shared backbone.

    Metadata input: 19 dimensions
      - age (1): normalized age
      - sex (4): one-hot encoded
      - site (7): one-hot encoded (reduced from 14)
      - MONET (7): 7 MONET feature scores
    """

    def __init__(
        self,
        model_name: str = 'convnext_base.fb_in22k_ft_in1k',
        metadata_dim: int = 19,
        num_classes: int = 11,
        dropout: float = 0.3
    ):
        super().__init__()

        self.backbone = timm.create_model(
            model_name,
            pretrained=False,
            num_classes=0
        )
        backbone_dim = self.backbone.num_features  # 1024 for convnext_base

        # Metadata MLP: 19 -> 64
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        # Classifier: 2112 -> 512 -> 256 -> 11
        # Input: clinical(1024) + derm(1024) + meta(64) = 2112
        fusion_dim = backbone_dim * 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

        self.metadata_dim = metadata_dim
        self.num_classes = num_classes
        self.backbone_dim = backbone_dim

    def forward(
        self,
        clinical_img: torch.Tensor,
        derm_img: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with dual images.

        Args:
            clinical_img: [B, 3, H, W] clinical image tensor
            derm_img: [B, 3, H, W] dermoscopy image tensor (uses clinical if None)
            metadata: [B, 19] metadata tensor (zeros if None)

        Returns:
            logits: [B, 11]
        """
        # Process clinical image
        clinical_features = self.backbone(clinical_img)

        # Process dermoscopy image
        if derm_img is not None:
            derm_features = self.backbone(derm_img)
        else:
            derm_features = clinical_features

        # Process metadata
        if metadata is not None:
            meta_features = self.meta_mlp(metadata)
        else:
            batch_size = clinical_features.size(0)
            meta_features = torch.zeros(
                batch_size, 64,
                device=clinical_features.device
            )

        # Concatenate: [B, 1024] + [B, 1024] + [B, 64] = [B, 2112]
        fused = torch.cat([clinical_features, derm_features, meta_features], dim=1)
        logits = self.classifier(fused)

        return logits


class ConvNeXtClassifier:
    """
    ConvNeXt classifier tool for skin lesion classification.
    Uses dual images (clinical + dermoscopy) and MONET features.
    """

    # Site mapping for metadata encoding
    SITE_MAPPING = {
        'head': 0, 'neck': 0, 'face': 0,  # head_neck_face
        'trunk': 1, 'back': 1, 'chest': 1, 'abdomen': 1,
        'upper': 2, 'arm': 2, 'hand': 2,  # upper extremity
        'lower': 3, 'leg': 3, 'foot': 3, 'thigh': 3,  # lower extremity
        'genital': 4, 'oral': 5, 'acral': 6,
    }

    SEX_MAPPING = {'male': 0, 'female': 1, 'other': 2, 'unknown': 3}

    def __init__(
        self,
        checkpoint_path: str = "models/seed42_fold0.pt",
        device: Optional[str] = None
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = None
        self.loaded = False

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load(self):
        """Load the ConvNeXt model from checkpoint"""
        if self.loaded:
            return

        # Determine device (respect SKINPRO_TOOL_DEVICE override for GPU sharing)
        forced = os.environ.get("SKINPRO_TOOL_DEVICE")
        if forced:
            self.device = forced
        elif self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        # Create model
        self.model = ConvNeXtDualEncoder(
            model_name='convnext_base.fb_in22k_ft_in1k',
            metadata_dim=19,
            num_classes=11,
            dropout=0.3
        )

        # Load checkpoint
        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False
        )

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()
        self.loaded = True

    def encode_metadata(
        self,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        site: Optional[str] = None,
        monet_scores: Optional[List[float]] = None
    ) -> torch.Tensor:
        """
        Encode metadata into 19-dim vector.

        Layout: [age(1), sex(4), site(7), monet(7)] = 19

        Args:
            age: Patient age in years
            sex: 'male', 'female', 'other', or None
            site: Anatomical site string
            monet_scores: List of 7 MONET feature scores

        Returns:
            torch.Tensor of shape [19]
        """
        features = []

        # Age (1 dim) - normalized
        age_norm = (age - 50) / 30 if age is not None else 0.0
        features.append(age_norm)

        # Sex (4 dim) - one-hot
        sex_onehot = [0.0] * 4
        if sex:
            sex_idx = self.SEX_MAPPING.get(sex.lower(), 3)
            sex_onehot[sex_idx] = 1.0
        features.extend(sex_onehot)

        # Site (7 dim) - one-hot
        site_onehot = [0.0] * 7
        if site:
            site_lower = site.lower()
            for key, idx in self.SITE_MAPPING.items():
                if key in site_lower:
                    site_onehot[idx] = 1.0
                    break
        features.extend(site_onehot)

        # MONET (7 dim)
        if monet_scores is not None and len(monet_scores) == 7:
            features.extend(monet_scores)
        else:
            features.extend([0.0] * 7)

        return torch.tensor(features, dtype=torch.float32)

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Preprocess PIL image for model input"""
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image).unsqueeze(0)

    def classify(
        self,
        clinical_image: Image.Image,
        derm_image: Optional[Image.Image] = None,
        age: Optional[float] = None,
        sex: Optional[str] = None,
        site: Optional[str] = None,
        monet_scores: Optional[List[float]] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Classify a skin lesion.

        Args:
            clinical_image: Clinical (close-up) image
            derm_image: Dermoscopy image (optional, uses clinical if None)
            age: Patient age
            sex: Patient sex
            site: Anatomical site
            monet_scores: 7 MONET feature scores
            top_k: Number of top predictions to return

        Returns:
            dict with 'predictions', 'probabilities', 'top_class', 'confidence'
        """
        if not self.loaded:
            self.load()

        # Preprocess images
        clinical_tensor = self.preprocess_image(clinical_image).to(self.device)

        if derm_image is not None:
            derm_tensor = self.preprocess_image(derm_image).to(self.device)
        else:
            derm_tensor = None

        # Encode metadata
        metadata = self.encode_metadata(age, sex, site, monet_scores)
        metadata_tensor = metadata.unsqueeze(0).to(self.device)

        # Run inference
        with torch.no_grad():
            logits = self.model(clinical_tensor, derm_tensor, metadata_tensor)
            probs = torch.softmax(logits, dim=1)[0].cpu().numpy()

        # Get top-k predictions
        top_indices = np.argsort(probs)[::-1][:top_k]

        predictions = []
        for idx in top_indices:
            predictions.append({
                'class': CLASS_NAMES[idx],
                'full_name': CLASS_FULL_NAMES[CLASS_NAMES[idx]],
                'probability': float(probs[idx])
            })

        return {
            'predictions': predictions,
            'probabilities': probs.tolist(),
            'top_class': CLASS_NAMES[top_indices[0]],
            'confidence': float(probs[top_indices[0]]),
            'all_classes': CLASS_NAMES,
        }

    def __call__(
        self,
        clinical_image: Image.Image,
        derm_image: Optional[Image.Image] = None,
        **kwargs
    ) -> Dict:
        """Shorthand for classify()"""
        return self.classify(clinical_image, derm_image, **kwargs)


# Singleton instance
_convnext_instance = None


def get_convnext_classifier(checkpoint_path: str = "models/seed42_fold0.pt") -> ConvNeXtClassifier:
    """Get or create ConvNeXt classifier instance"""
    global _convnext_instance
    if _convnext_instance is None:
        _convnext_instance = ConvNeXtClassifier(checkpoint_path)
    return _convnext_instance


if __name__ == "__main__":
    import sys

    print("ConvNeXt Classifier Test")
    print("=" * 50)

    classifier = ConvNeXtClassifier()
    print("Loading model...")
    classifier.load()
    print("Model loaded!")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nClassifying: {image_path}")

        image = Image.open(image_path).convert("RGB")

        # Example with mock MONET scores
        monet_scores = [0.2, 0.1, 0.05, 0.3, 0.7, 0.1, 0.05]

        result = classifier.classify(
            clinical_image=image,
            age=55,
            sex="male",
            site="back",
            monet_scores=monet_scores
        )

        print("\nTop Predictions:")
        for pred in result['predictions']:
            print(f"  {pred['probability']:.1%} - {pred['class']} ({pred['full_name']})")
