# AI-generated
"""
encoder.py — MobileNetV3-Small backbone with 128-dim L2-normalized embedding head.

The encoder is shared between the drone (bird's eye) branch and the satellite branch.
Weights are loaded from ImageNet pretraining and fine-tuned end-to-end during contrastive
training. At inference the final layer is L2-normalized so cosine similarity == dot product,
which lets FAISS use an exact inner-product (IP) index without a separate normalization step.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class Encoder(nn.Module):
    """
    MobileNetV3-Small backbone → 128-dim L2-normalized embedding.

    Args:
        embedding_dim: Output dimensionality (default 128).
        pretrained: Load ImageNet weights for the backbone (default True).
        freeze_backbone: If True, only the projection head is trained. Useful
                         for a first warm-up pass when data is scarce.
    """

    def __init__(
        self,
        embedding_dim: int = 128,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.mobilenet_v3_small(weights=weights)

        # Drop the original classifier; keep only the feature extractor + adaptive pool.
        # MobileNetV3-Small produces 576-dim features after adaptive avg pool.
        self.features = backbone.features
        self.pool = backbone.avgpool  # AdaptiveAvgPool2d(1)
        backbone_out_dim = 576

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        # Lightweight projection head: Linear → BN → ReLU → Linear
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_out_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Linear(256, embedding_dim),
        )

        self.embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 3, 224, 224) — images pre-normalised to ImageNet stats.
        Returns:
            (B, embedding_dim) — L2-normalised embeddings.
        """
        x = self.features(x)
        x = self.pool(x)
        x = self.head(x)
        x = nn.functional.normalize(x, p=2, dim=1)
        return x
