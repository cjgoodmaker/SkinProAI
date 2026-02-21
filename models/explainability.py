# models/explainability.py

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from typing import Tuple
from PIL import Image

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping
    Shows which regions of image are important for prediction
    """
    
    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        """
        Args:
            model: The neural network
            target_layer: Layer name to compute CAM on (usually last conv layer)
        """
        self.model = model
        self.gradients = None
        self.activations = None
        
        # Auto-detect target layer if not specified
        if target_layer is None:
            # Use last ConvNeXt stage
            self.target_layer = model.convnext.stages[-1]
        else:
            self.target_layer = dict(model.named_modules())[target_layer]
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save forward activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(
        self,
        image: torch.Tensor,
        target_class: int = None
    ) -> np.ndarray:
        """
        Generate Class Activation Map
        
        Args:
            image: Input image [1, 3, H, W]
            target_class: Class to generate CAM for (None = predicted class)
            
        Returns:
            cam: Activation map [H, W] normalized to 0-1
        """
        self.model.eval()
        
        # Forward pass
        output = self.model(image)
        
        # Use predicted class if not specified
        if target_class is None:
            target_class = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Backward pass for target class
        output[0, target_class].backward()
        
        # Get gradients and activations
        gradients = self.gradients[0]  # [C, H, W]
        activations = self.activations[0]  # [C, H, W]
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2))  # [C]
        
        # Weighted sum of activations
        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU
        cam = F.relu(cam)
        
        # Normalize to 0-1
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        
        return cam
    
    def overlay_cam_on_image(
        self,
        image: np.ndarray,  # [H, W, 3] RGB
        cam: np.ndarray,    # [h, w]
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET
    ) -> np.ndarray:
        """
        Overlay CAM heatmap on original image
        
        Returns:
            overlay: [H, W, 3] RGB image with heatmap
        """
        H, W = image.shape[:2]
        
        # Resize CAM to image size
        cam_resized = cv2.resize(cam, (W, H))
        
        # Convert to heatmap
        heatmap = cv2.applyColorMap(
            np.uint8(255 * cam_resized),
            colormap
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend with original image
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlay

class AttentionVisualizer:
    """Visualize MedSigLIP attention maps"""
    
    def __init__(self, model):
        self.model = model
    
    def get_attention_maps(self, image: torch.Tensor) -> np.ndarray:
        """
        Extract attention maps from MedSigLIP
        
        Returns:
            attention: [num_heads, H, W] attention weights
        """
        # Forward pass
        with torch.no_grad():
            _ = self.model(image)
        
        # Get last layer attention from MedSigLIP
        # Shape: [batch, num_heads, seq_len, seq_len]
        attention = self.model.medsiglip_features
        
        # Average across heads and extract spatial attention
        # This is model-dependent - adjust based on MedSigLIP architecture
        
        # Placeholder implementation
        # You'll need to adapt this to your specific MedSigLIP implementation
        return np.random.rand(14, 14)  # Placeholder
    
    def overlay_attention(
        self,
        image: np.ndarray,
        attention: np.ndarray,
        alpha: float = 0.6
    ) -> np.ndarray:
        """Overlay attention map on image"""
        H, W = image.shape[:2]
        
        # Resize attention to image size
        attention_resized = cv2.resize(attention, (W, H))
        
        # Normalize
        attention_resized = (attention_resized - attention_resized.min())
        if attention_resized.max() > 0:
            attention_resized = attention_resized / attention_resized.max()
        
        # Create colored overlay
        heatmap = cv2.applyColorMap(
            np.uint8(255 * attention_resized),
            cv2.COLORMAP_VIRIDIS
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = (alpha * heatmap + (1 - alpha) * image).astype(np.uint8)
        
        return overlay