# AI-generated
"""
infonce.py — InfoNCE (NT-Xent) contrastive loss.

Given a batch of N positive pairs (q_i, k_i), the loss encourages each query
embedding q_i to be closer to its key k_i than to all other keys in the batch.

    L = -1/N * sum_i log( exp(q_i·k_i / τ) / sum_j exp(q_i·k_j / τ) )

References:
    Oord et al., "Representation Learning with Contrastive Predictive Coding" (2018)
    Chen et al., "A Simple Framework for Contrastive Learning" (SimCLR, 2020)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """
    Symmetric InfoNCE loss for a batch of positive pairs.

    Both (query→key) and (key→query) directions are averaged, which is
    equivalent to treating every sample as both a query and a key.

    Args:
        temperature: Softmax temperature τ. Lower values make the distribution
                     sharper and the task harder. Typical range: 0.05 – 0.2.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}")
        self.temperature = temperature

    def forward(
        self, query: torch.Tensor, key: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            query: (B, D) — L2-normalised embeddings from the drone branch.
            key:   (B, D) — L2-normalised embeddings from the satellite branch.
                   query[i] and key[i] are a positive pair.
        Returns:
            Scalar loss tensor.
        """
        B = query.size(0)
        if B < 2:
            raise ValueError(
                "InfoNCE requires batch size >= 2 (need at least one negative). "
                f"Got batch size {B}."
            )

        # Similarity matrix: (B, B), entry [i,j] = q_i · k_j / τ
        logits = torch.mm(query, key.T) / self.temperature   # (B, B)

        # Targets: diagonal entries are positives
        labels = torch.arange(B, device=query.device)

        # Symmetric: average loss in both directions
        loss_q2k = F.cross_entropy(logits, labels)
        loss_k2q = F.cross_entropy(logits.T, labels)

        return (loss_q2k + loss_k2q) / 2
