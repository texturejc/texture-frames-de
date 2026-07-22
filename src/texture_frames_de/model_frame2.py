"""
Frame classification v2 — marker-token-pooled classifier.

DeBERTa backbone; the frame representation is the concatenation of the <t> and
</t> marker tokens' hidden states (predicate-focused) rather than [CLS]. This
targets the frame discrimination gap that soft-masking and candidate-name
conditioning couldn't (0.863 -> ?; baseline 0.887).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class FrameMarkerModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_frames: int, dropout: float = 0.1):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(2 * h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_frames),
        )
        self.num_frames = num_frames

    @classmethod
    def from_pretrained(cls, base_model: str, num_frames: int, dropout: float = 0.1,
                        torch_dtype=torch.float32) -> "FrameMarkerModel":
        backbone = AutoModel.from_pretrained(base_model, torch_dtype=torch_dtype)
        return cls(backbone, num_frames, dropout=dropout)

    def resize_token_embeddings(self, n: int):
        self.backbone.resize_token_embeddings(n)

    def encode_logits(self, input_ids, attention_mask, start_pos, end_pos):
        """Frame logits (B, num_frames) from the two marker tokens' hidden states."""
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        ar = torch.arange(hidden.size(0), device=hidden.device)
        rep = torch.cat([hidden[ar, start_pos], hidden[ar, end_pos]], dim=-1)
        return self.classifier(rep)

    def forward(self, input_ids, attention_mask, start_pos, end_pos, labels=None, **_):
        logits = self.encode_logits(input_ids, attention_mask, start_pos, end_pos)
        loss = None if labels is None else F.cross_entropy(logits, labels)
        return {"loss": loss, "logits": logits}
