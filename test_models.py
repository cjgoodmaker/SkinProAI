#!/usr/bin/env python3
"""Test script to verify model loading"""

import torch
import torch.nn as nn
import timm
from transformers import AutoModel, AutoProcessor
import numpy as np

DEVICE = "cpu"
print(f"Device: {DEVICE}")

# ConvNeXt model definition (matching checkpoint)
class ConvNeXtDualEncoder(nn.Module):
    def __init__(self, model_name="convnext_base.fb_in22k_ft_in1k",
                 metadata_dim=19, num_classes=11, dropout=0.3):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=False, num_classes=0)
        backbone_dim = self.backbone.num_features
        self.meta_mlp = nn.Sequential(
            nn.Linear(metadata_dim, 64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(dropout)
        )
        fusion_dim = backbone_dim * 2 + 64
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )

    def forward(self, clinical_img, derm_img=None, metadata=None):
        clinical_features = self.backbone(clinical_img)
        derm_features = self.backbone(derm_img) if derm_img is not None else clinical_features
        if metadata is not None:
            meta_features = self.meta_mlp(metadata)
        else:
            meta_features = torch.zeros(clinical_features.size(0), 64, device=clinical_features.device)
        fused = torch.cat([clinical_features, derm_features, meta_features], dim=1)
        return self.classifier(fused)


# MedSigLIP model definition
class MedSigLIPClassifier(nn.Module):
    def __init__(self, num_classes=11, model_name="google/siglip-base-patch16-384"):
        super().__init__()
        self.siglip = AutoModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        hidden_dim = self.siglip.config.vision_config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 512), nn.LayerNorm(512), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        for param in self.siglip.parameters():
            param.requires_grad = False

    def forward(self, pixel_values):
        vision_outputs = self.siglip.vision_model(pixel_values=pixel_values)
        pooled_features = vision_outputs.pooler_output
        return self.classifier(pooled_features)


if __name__ == "__main__":
    print("\n[1/2] Loading ConvNeXt...")
    convnext_model = ConvNeXtDualEncoder()
    ckpt = torch.load("models/seed42_fold0.pt", map_location=DEVICE, weights_only=False)
    convnext_model.load_state_dict(ckpt)
    convnext_model.eval()
    print("   ConvNeXt loaded!")

    print("\n[2/2] Loading MedSigLIP...")
    medsiglip_model = MedSigLIPClassifier()
    medsiglip_model.eval()
    print("   MedSigLIP loaded!")

    # Quick inference test
    print("\nTesting inference...")
    dummy_img = torch.randn(1, 3, 384, 384)
    with torch.no_grad():
        convnext_out = convnext_model(dummy_img)
        print(f"   ConvNeXt output: {convnext_out.shape}")

        dummy_pil = np.random.randint(0, 255, (384, 384, 3), dtype=np.uint8)
        siglip_input = medsiglip_model.processor(images=[dummy_pil], return_tensors="pt")
        siglip_out = medsiglip_model(siglip_input["pixel_values"])
        print(f"   MedSigLIP output: {siglip_out.shape}")

    print("\nAll tests passed!")
