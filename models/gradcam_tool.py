"""
Grad-CAM Tool - Visual explanation of ConvNeXt predictions
Shows which regions of the image the model focuses on.
"""

import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torchvision import transforms
from typing import Optional, Tuple
import cv2


class GradCAM:
    """
    Grad-CAM implementation for ConvNeXt model.
    Generates heatmaps showing model attention.
    """

    def __init__(self, model, target_layer=None):
        """
        Args:
            model: ConvNeXtDualEncoder model
            target_layer: Layer to extract gradients from (default: last conv layer)
        """
        self.model = model
        self.gradients = None
        self.activations = None

        # Hook the target layer (last stage of backbone)
        if target_layer is None:
            target_layer = model.backbone.stages[-1]

        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """Save activations during forward pass"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """Save gradients during backward pass"""
        self.gradients = grad_output[0].detach()

    def generate(
        self,
        image_tensor: torch.Tensor,
        target_class: Optional[int] = None,
        derm_tensor: Optional[torch.Tensor] = None,
        metadata: Optional[torch.Tensor] = None
    ) -> np.ndarray:
        """
        Generate Grad-CAM heatmap.

        Args:
            image_tensor: Input image tensor [1, 3, H, W]
            target_class: Class index to visualize (default: predicted class)
            derm_tensor: Optional dermoscopy image tensor
            metadata: Optional metadata tensor

        Returns:
            CAM heatmap as numpy array [H, W] normalized to 0-1
        """
        self.model.eval()

        # Forward pass
        output = self.model(image_tensor, derm_tensor, metadata)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        self.model.zero_grad()
        output[0, target_class].backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]

        # Weighted combination of activation maps
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32, device=activations.device)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()

        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())

        return cam

    def overlay(
        self,
        image: np.ndarray,
        cam: np.ndarray,
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on original image.

        Args:
            image: Original image [H, W, 3] RGB uint8
            cam: CAM heatmap [H, W] float 0-1
            alpha: Overlay transparency
            colormap: OpenCV colormap

        Returns:
            Overlaid image [H, W, 3] RGB uint8
        """
        H, W = image.shape[:2]

        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (W, H))

        # Apply colormap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)

        return overlay


class GradCAMTool:
    """
    High-level Grad-CAM tool for ConvNeXt classifier.
    """

    def __init__(self, classifier=None):
        """
        Args:
            classifier: ConvNeXtClassifier instance (will create one if None)
        """
        self.classifier = classifier
        self.gradcam = None
        self.loaded = False

        # Preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def load(self):
        """Load classifier and setup Grad-CAM"""
        if self.loaded:
            return

        if self.classifier is None:
            from models.convnext_classifier import ConvNeXtClassifier
            self.classifier = ConvNeXtClassifier()
            self.classifier.load()

        self.gradcam = GradCAM(self.classifier.model)
        self.loaded = True

    def generate_heatmap(
        self,
        image: Image.Image,
        target_class: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, int, float]:
        """
        Generate Grad-CAM heatmap for an image.

        Args:
            image: PIL Image
            target_class: Class to visualize (default: predicted)

        Returns:
            Tuple of (overlay_image, cam_heatmap, predicted_class, confidence)
        """
        if not self.loaded:
            self.load()

        # Ensure RGB
        if image.mode != "RGB":
            image = image.convert("RGB")

        # Preprocess
        image_np = np.array(image.resize((384, 384)))
        image_tensor = self.transform(image).unsqueeze(0).to(self.classifier.device)

        # Get prediction first
        with torch.no_grad():
            logits = self.classifier.model(image_tensor)
            probs = torch.softmax(logits, dim=1)[0]
            pred_class = probs.argmax().item()
            confidence = probs[pred_class].item()

        # Use predicted class if not specified
        if target_class is None:
            target_class = pred_class

        # Generate CAM
        cam = self.gradcam.generate(image_tensor, target_class)

        # Create overlay
        overlay = self.gradcam.overlay(image_np, cam, alpha=0.5)

        return overlay, cam, pred_class, confidence

    def analyze(
        self,
        image: Image.Image,
        target_class: Optional[int] = None
    ) -> dict:
        """
        Full analysis with Grad-CAM visualization.

        Args:
            image: PIL Image
            target_class: Class to visualize

        Returns:
            Dict with overlay_image, cam, prediction info
        """
        from models.convnext_classifier import CLASS_NAMES, CLASS_FULL_NAMES

        overlay, cam, pred_class, confidence = self.generate_heatmap(image, target_class)

        return {
            "overlay": Image.fromarray(overlay),
            "cam": cam,
            "predicted_class": CLASS_NAMES[pred_class],
            "predicted_class_full": CLASS_FULL_NAMES[CLASS_NAMES[pred_class]],
            "confidence": confidence,
            "class_index": pred_class,
        }

    def __call__(self, image: Image.Image, target_class: Optional[int] = None) -> dict:
        return self.analyze(image, target_class)


# Singleton
_gradcam_instance = None


def get_gradcam_tool() -> GradCAMTool:
    """Get or create Grad-CAM tool instance"""
    global _gradcam_instance
    if _gradcam_instance is None:
        _gradcam_instance = GradCAMTool()
    return _gradcam_instance


if __name__ == "__main__":
    import sys

    print("Grad-CAM Tool Test")
    print("=" * 50)

    tool = GradCAMTool()
    print("Loading model...")
    tool.load()
    print("Model loaded!")

    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        print(f"\nAnalyzing: {image_path}")

        image = Image.open(image_path).convert("RGB")
        result = tool.analyze(image)

        print(f"\nPrediction: {result['predicted_class']} ({result['confidence']:.1%})")
        print(f"Full name: {result['predicted_class_full']}")

        # Save overlay
        output_path = image_path.rsplit(".", 1)[0] + "_gradcam.png"
        result["overlay"].save(output_path)
        print(f"\nGrad-CAM overlay saved to: {output_path}")
