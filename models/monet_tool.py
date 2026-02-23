"""
MONET Tool - Skin lesion feature extraction using MONET model
Correct implementation based on MONET tutorial: automatic_concept_annotation.ipynb
"""

import os
import torch
import torch.nn.functional as F
import numpy as np
import scipy.special
from PIL import Image
from typing import Optional, Dict, List
import torchvision.transforms as T


# The 7 MONET feature columns expected by ConvNeXt
MONET_FEATURES = [
    "MONET_ulceration_crust",
    "MONET_hair",
    "MONET_vasculature_vessels",
    "MONET_erythema",
    "MONET_pigmented",
    "MONET_gel_water_drop_fluid_dermoscopy_liquid",
    "MONET_skin_markings_pen_ink_purple_pen",
]

# Concept terms for each MONET feature (multiple synonyms improve detection)
MONET_CONCEPT_TERMS = {
    "MONET_ulceration_crust": ["ulceration", "crust", "crusting", "ulcer"],
    "MONET_hair": ["hair", "hairy"],
    "MONET_vasculature_vessels": ["blood vessels", "vasculature", "vessels", "telangiectasia"],
    "MONET_erythema": ["erythema", "redness", "red"],
    "MONET_pigmented": ["pigmented", "pigmentation", "melanin", "brown"],
    "MONET_gel_water_drop_fluid_dermoscopy_liquid": ["dermoscopy gel", "fluid", "water drop", "immersion fluid"],
    "MONET_skin_markings_pen_ink_purple_pen": ["pen marking", "ink", "surgical marking", "purple pen"],
}

# Prompt templates (from MONET paper)
PROMPT_TEMPLATES = [
    "This is skin image of {}",
    "This is dermatology image of {}",
    "This is image of {}",
]

# Reference prompts (baseline for contrastive scoring)
PROMPT_REFS = [
    ["This is skin image"],
    ["This is dermatology image"],
    ["This is image"],
]


def get_transform(n_px=224):
    """Get MONET preprocessing transform"""
    def convert_image_to_rgb(image):
        return image.convert("RGB")

    return T.Compose([
        T.Resize(n_px, interpolation=T.InterpolationMode.BICUBIC),
        T.CenterCrop(n_px),
        convert_image_to_rgb,
        T.ToTensor(),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])


class MonetTool:
    """
    MONET tool for extracting concept presence scores from skin lesion images.
    Uses the proper contrastive scoring method from the MONET paper.
    """

    def __init__(self, device: Optional[str] = None, use_hf: bool = True):
        """
        Args:
            device: Device to run on (cuda, mps, cpu)
            use_hf: Use HuggingFace implementation (True) or original CLIP (False)
        """
        self.model = None
        self.processor = None
        self.device = device
        self.use_hf = use_hf
        self.loaded = False
        self.transform = get_transform(224)

        # Cache for concept embeddings
        self._concept_embeddings = {}

    def load(self):
        """Load MONET model"""
        if self.loaded:
            return

        # Determine device (respect SKINPRO_TOOL_DEVICE override for GPU sharing)
        forced = os.environ.get("SKINPRO_TOOL_DEVICE")
        if forced:
            self.device = forced
        elif self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda:0"
            elif torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        if self.use_hf:
            # HuggingFace implementation
            from transformers import AutoProcessor, AutoModelForZeroShotImageClassification

            self.processor = AutoProcessor.from_pretrained("chanwkim/monet")
            self.model = AutoModelForZeroShotImageClassification.from_pretrained("chanwkim/monet")
            self.model.to(self.device)
            self.model.eval()
        else:
            # Original CLIP implementation
            import clip

            self.model, _ = clip.load("ViT-L/14", device=self.device, jit=False)
            self.model.load_state_dict(
                torch.hub.load_state_dict_from_url(
                    "https://aimslab.cs.washington.edu/MONET/weight_clip.pt"
                )
            )
            self.model.eval()

        self.loaded = True

        # Pre-compute concept embeddings for all MONET features
        self._precompute_concept_embeddings()

    def _encode_text(self, text_list: List[str]) -> torch.Tensor:
        """Encode text to embeddings"""
        if self.use_hf:
            inputs = self.processor(text=text_list, return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.no_grad():
                embeddings = self.model.get_text_features(**inputs)
        else:
            import clip
            tokens = clip.tokenize(text_list, truncate=True).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_text(tokens)

        return embeddings.cpu()

    def _encode_image(self, image: Image.Image) -> torch.Tensor:
        """Encode image to embedding"""
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        if self.use_hf:
            with torch.no_grad():
                embedding = self.model.get_image_features(image_tensor)
        else:
            with torch.no_grad():
                embedding = self.model.encode_image(image_tensor)

        return embedding.cpu()

    def _precompute_concept_embeddings(self):
        """Pre-compute embeddings for all MONET concepts"""
        for feature_name, concept_terms in MONET_CONCEPT_TERMS.items():
            self._concept_embeddings[feature_name] = self._get_concept_embedding(concept_terms)

    def _get_concept_embedding(self, concept_terms: List[str]) -> Dict:
        """
        Generate prompt embeddings for a concept using multiple templates.

        Args:
            concept_terms: List of synonymous terms for the concept

        Returns:
            dict with target and reference embeddings
        """
        # Target prompts: "This is skin image of {term}"
        prompt_target = [
            [template.format(term) for term in concept_terms]
            for template in PROMPT_TEMPLATES
        ]

        # Flatten and encode
        prompt_target_flat = [p for template_prompts in prompt_target for p in template_prompts]
        target_embeddings = self._encode_text(prompt_target_flat)

        # Reshape to [num_templates, num_terms, embed_dim]
        num_templates = len(PROMPT_TEMPLATES)
        num_terms = len(concept_terms)
        embed_dim = target_embeddings.shape[-1]
        target_embeddings = target_embeddings.view(num_templates, num_terms, embed_dim)

        # Normalize
        target_embeddings_norm = F.normalize(target_embeddings, dim=2)

        # Reference prompts: "This is skin image"
        prompt_ref_flat = [p for ref_list in PROMPT_REFS for p in ref_list]
        ref_embeddings = self._encode_text(prompt_ref_flat)
        ref_embeddings = ref_embeddings.view(num_templates, -1, embed_dim)
        ref_embeddings_norm = F.normalize(ref_embeddings, dim=2)

        return {
            "target_embedding_norm": target_embeddings_norm,
            "ref_embedding_norm": ref_embeddings_norm,
        }

    def _calculate_concept_score(
        self,
        image_features_norm: torch.Tensor,
        concept_embedding: Dict,
        temp: float = 1 / np.exp(4.5944)
    ) -> float:
        """
        Calculate concept presence score using contrastive comparison.

        Args:
            image_features_norm: Normalized image embedding [1, embed_dim]
            concept_embedding: Dict with target and reference embeddings
            temp: Temperature for softmax

        Returns:
            Concept presence score (0-1)
        """
        target_emb = concept_embedding["target_embedding_norm"].float()
        ref_emb = concept_embedding["ref_embedding_norm"].float()

        # Similarity: [num_templates, num_terms] @ [embed_dim, 1] -> [num_templates, num_terms, 1]
        target_similarity = target_emb @ image_features_norm.T.float()
        ref_similarity = ref_emb @ image_features_norm.T.float()

        # Mean over terms for each template
        target_mean = target_similarity.mean(dim=1).squeeze()  # [num_templates]
        ref_mean = ref_similarity.mean(dim=1).squeeze()  # [num_templates]

        # Softmax between target and reference (contrastive scoring)
        scores = scipy.special.softmax(
            np.array([target_mean.numpy() / temp, ref_mean.numpy() / temp]),
            axis=0
        )

        # Return mean of target scores across templates
        return float(scores[0].mean())

    def extract_features(self, image: Image.Image) -> Dict[str, float]:
        """
        Extract MONET feature scores from a skin lesion image.

        Args:
            image: PIL Image to analyze

        Returns:
            dict with 7 MONET feature scores (0-1 range)
        """
        if not self.loaded:
            self.load()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Get image embedding
        image_features = self._encode_image(image)
        image_features_norm = F.normalize(image_features, dim=1)

        # Calculate score for each MONET feature
        features = {}
        for feature_name in MONET_FEATURES:
            concept_emb = self._concept_embeddings[feature_name]
            score = self._calculate_concept_score(image_features_norm, concept_emb)
            features[feature_name] = score

        return features

    def get_feature_vector(self, image: Image.Image) -> List[float]:
        """Get MONET features as a list in the expected order."""
        features = self.extract_features(image)
        return [features[f] for f in MONET_FEATURES]

    def get_feature_tensor(self, image: Image.Image) -> torch.Tensor:
        """Get MONET features as a PyTorch tensor."""
        return torch.tensor(self.get_feature_vector(image), dtype=torch.float32)

    def describe_features(self, features: Dict[str, float], threshold: float = 0.6) -> str:
        """Generate a natural language description of the MONET features."""
        descriptions = {
            "MONET_ulceration_crust": "ulceration or crusting",
            "MONET_hair": "visible hair",
            "MONET_vasculature_vessels": "visible blood vessels",
            "MONET_erythema": "erythema (redness)",
            "MONET_pigmented": "pigmentation",
            "MONET_gel_water_drop_fluid_dermoscopy_liquid": "dermoscopy gel/fluid",
            "MONET_skin_markings_pen_ink_purple_pen": "pen markings",
        }

        present = []
        for feature, score in features.items():
            if score >= threshold:
                desc = descriptions.get(feature, feature)
                present.append(f"{desc} ({score:.0%})")

        if not present:
            # Show top features even if below threshold
            sorted_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:3]
            present = [f"{descriptions.get(f, f)} ({s:.0%})" for f, s in sorted_features]

        return "Detected features: " + ", ".join(present)

    def analyze(self, image: Image.Image) -> Dict:
        """Full analysis returning features, vector, and description."""
        features = self.extract_features(image)
        vector = [features[f] for f in MONET_FEATURES]
        description = self.describe_features(features)

        return {
            "features": features,
            "vector": vector,
            "description": description,
            "feature_names": MONET_FEATURES,
        }

    def __call__(self, image: Image.Image) -> Dict:
        """Shorthand for analyze()"""
        return self.analyze(image)


# Singleton instance
_monet_instance = None


def get_monet_tool() -> MonetTool:
    """Get or create MONET tool instance"""
    global _monet_instance
    if _monet_instance is None:
        _monet_instance = MonetTool()
    return _monet_instance


if __name__ == "__main__":
    import sys

    print("MONET Tool Test (Correct Implementation)")
    print("=" * 50)

    tool = MonetTool(use_hf=True)
    print("Loading model...")
    tool.load()
    print("Model loaded!")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nAnalyzing: {image_path}")
        image = Image.open(image_path).convert("RGB")
        result = tool.analyze(image)

        print("\nMONET Features (Contrastive Scores):")
        for name, score in result["features"].items():
            bar = "█" * int(score * 20)
            print(f"  {name}: {score:.3f} {bar}")

        print(f"\nDescription: {result['description']}")
        print(f"\nVector: {[f'{v:.3f}' for v in result['vector']]}")
