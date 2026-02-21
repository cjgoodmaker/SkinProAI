# models/medsiglip_convnext_fusion.py

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import numpy as np
import timm
from transformers import AutoModel, AutoProcessor

class MedSigLIPConvNeXtFusion(nn.Module):
    """
    Your trained MedSigLIP-ConvNeXt fusion model from MILK10 challenge
    Supports 11-class skin lesion classification
    """
    
    # Class names from your training
    CLASS_NAMES = [
        'AKIEC',      # Actinic Keratoses and Intraepithelial Carcinoma
        'BCC',        # Basal Cell Carcinoma
        'BEN_OTH',    # Benign Other
        'BKL',        # Benign Keratosis-like Lesions
        'DF',         # Dermatofibroma
        'INF',        # Inflammatory
        'MAL_OTH',    # Malignant Other
        'MEL',        # Melanoma
        'NV',         # Melanocytic Nevi
        'SCCKA',      # Squamous Cell Carcinoma and Keratoacanthoma
        'VASC'        # Vascular Lesions
    ]
    
    def __init__(
        self,
        num_classes: int = 11,
        medsiglip_model: str = "google/medsiglip-base",
        convnext_variant: str = "convnext_base",
        fusion_dim: int = 512,
        dropout: float = 0.3,
        metadata_dim: int = 20  # For metadata features
    ):
        super().__init__()
        
        self.num_classes = num_classes
        
        # MedSigLIP Vision Encoder
        print(f"Loading MedSigLIP: {medsiglip_model}")
        self.medsiglip = AutoModel.from_pretrained(medsiglip_model)
        self.medsiglip_processor = AutoProcessor.from_pretrained(medsiglip_model)
        
        # ConvNeXt Backbone
        print(f"Loading ConvNeXt: {convnext_variant}")
        self.convnext = timm.create_model(
            convnext_variant,
            pretrained=True,
            num_classes=0,
            global_pool='avg'
        )
        
        # Feature dimensions
        self.medsiglip_dim = self.medsiglip.config.hidden_size  # 768
        self.convnext_dim = self.convnext.num_features  # 1024
        
        # Optional metadata branch
        self.use_metadata = metadata_dim > 0
        if self.use_metadata:
            self.metadata_encoder = nn.Sequential(
                nn.Linear(metadata_dim, 64),
                nn.LayerNorm(64),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(64, 32)
            )
            total_dim = self.medsiglip_dim + self.convnext_dim + 32
        else:
            total_dim = self.medsiglip_dim + self.convnext_dim
        
        # Fusion layers
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.LayerNorm(fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        
        # Store intermediate features for Grad-CAM
        self.convnext_features = None
        self.medsiglip_features = None
        
        # Register hooks
        self.convnext.stages[-1].register_forward_hook(self._save_convnext_features)
        
    def _save_convnext_features(self, module, input, output):
        """Hook to save ConvNeXt feature maps for Grad-CAM"""
        self.convnext_features = output
    
    def forward(
        self, 
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            image: [B, 3, H, W] tensor
            metadata: [B, metadata_dim] optional metadata features
            
        Returns:
            logits: [B, num_classes]
        """
        # MedSigLIP features
        medsiglip_out = self.medsiglip.vision_model(image)
        medsiglip_features = medsiglip_out.pooler_output  # [B, 768]
        
        # ConvNeXt features  
        convnext_features = self.convnext(image)  # [B, 1024]
        
        # Concatenate vision features
        fused = torch.cat([medsiglip_features, convnext_features], dim=1)
        
        # Add metadata if available
        if self.use_metadata and metadata is not None:
            metadata_features = self.metadata_encoder(metadata)
            fused = torch.cat([fused, metadata_features], dim=1)
        
        # Fusion layers
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits
    
    def predict(
        self,
        image: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        top_k: int = 5
    ) -> Dict:
        """
        Get predictions with probabilities
        
        Args:
            image: [B, 3, H, W] or [3, H, W]
            metadata: Optional metadata features
            top_k: Number of top predictions
            
        Returns:
            Dictionary with predictions and features
        """
        if image.dim() == 3:
            image = image.unsqueeze(0)
        
        self.eval()
        with torch.no_grad():
            logits = self.forward(image, metadata)
            probs = torch.softmax(logits, dim=1)
            
            # Top-k predictions
            top_probs, top_indices = torch.topk(
                probs, 
                k=min(top_k, self.num_classes), 
                dim=1
            )
            
            # Format results
            predictions = []
            for i in range(top_probs.size(1)):
                predictions.append({
                    'class': self.CLASS_NAMES[top_indices[0, i].item()],
                    'probability': top_probs[0, i].item(),
                    'class_idx': top_indices[0, i].item()
                })
        
        return {
            'predictions': predictions,
            'all_probabilities': probs[0].cpu().numpy(),
            'logits': logits[0].cpu().numpy(),
            'convnext_features': self.convnext_features,
            'medsiglip_features': self.medsiglip_features
        }
    
    @classmethod
    def load_from_checkpoint(
        cls,
        medsiglip_path: str,
        convnext_path: Optional[str] = None,
        ensemble_weights: tuple = (0.6, 0.4),
        device: str = 'cpu'
    ):
        """
        Load model from your training checkpoints
        
        Args:
            medsiglip_path: Path to MedSigLIP model weights
            convnext_path: Path to ConvNeXt model weights (optional)
            ensemble_weights: (w_medsiglip, w_convnext) 
            device: Device to load on
        """
        model = cls(num_classes=11)
        
        # Load MedSigLIP weights
        print(f"Loading MedSigLIP from: {medsiglip_path}")
        medsiglip_state = torch.load(medsiglip_path, map_location=device)
        
        # Handle different checkpoint formats
        if 'model_state_dict' in medsiglip_state:
            model.load_state_dict(medsiglip_state['model_state_dict'])
        else:
            model.load_state_dict(medsiglip_state)
        
        # Store ensemble weights for prediction fusion
        model.ensemble_weights = ensemble_weights
        
        model.to(device)
        model.eval()
        
        return model