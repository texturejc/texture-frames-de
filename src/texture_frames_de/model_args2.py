"""
Argument extraction v2 — the detect-then-classify model.

One DeBERTa backbone, two heads (ARGS_V2_DESIGN.md):
  * detect_head  — 3-class BIO over tokens (O/B/I), "is this token part of an
    argument span" (role-agnostic).
  * role_head    — per span: pool the span's tokens (start ⊕ end ⊕ mean) and
    classify into NULL + the FE vocab. Trained on gold spans + sampled NULL
    negatives; at inference it labels the spans that detect_head produced.

Single forward pass, so the encoder speed advantage over the generative baseline
is preserved. Training loss = detect_CE + λ·role_CE.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

IGNORE_INDEX = -100


class Args2Model(nn.Module):
    def __init__(
        self,
        backbone: nn.Module,
        num_roles: int,
        role_lambda: float = 1.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        h = backbone.config.hidden_size
        self.detect_head = nn.Linear(h, 3)
        self.role_head = nn.Sequential(
            nn.Linear(3 * h, h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(h, num_roles),
        )
        self.num_roles = num_roles
        self.role_lambda = role_lambda

    @classmethod
    def from_pretrained(
        cls,
        base_model: str,
        num_roles: int,
        role_lambda: float = 1.0,
        dropout: float = 0.1,
        torch_dtype=torch.float32,
    ) -> "Args2Model":
        backbone = AutoModel.from_pretrained(base_model, torch_dtype=torch_dtype)
        return cls(backbone, num_roles, role_lambda=role_lambda, dropout=dropout)

    def resize_token_embeddings(self, n: int):
        self.backbone.resize_token_embeddings(n)

    # -- encoding + span pooling ------------------------------------------- #
    def encode(self, input_ids, attention_mask):
        """Return (hidden_states (B,T,H), detect_logits (B,T,3))."""
        hidden = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask
        ).last_hidden_state
        return hidden, self.detect_head(hidden)

    def _span_reps(self, hidden, span_bi: list[tuple[int, int, int]]) -> torch.Tensor:
        """span_bi: [(batch_idx, start_tok, end_tok_inclusive), ...] -> (N, 3H)."""
        reps = []
        for b, s, e in span_bi:
            h = hidden[b]
            reps.append(torch.cat([h[s], h[e], h[s : e + 1].mean(dim=0)], dim=-1))
        if not reps:
            return hidden.new_zeros((0, 3 * hidden.size(-1)))
        return torch.stack(reps)

    def role_logits_for_spans(self, hidden, span_bi: list[tuple[int, int, int]]) -> torch.Tensor:
        """Role logits (N, num_roles) for the given spans — used at inference."""
        return self.role_head(self._span_reps(hidden, span_bi))

    # -- training forward -------------------------------------------------- #
    def forward(self, input_ids, attention_mask, detect_labels=None, spans=None, **_):
        hidden, detect_logits = self.encode(input_ids, attention_mask)

        if detect_labels is None:
            return {"detect_logits": detect_logits}

        detect_loss = F.cross_entropy(
            detect_logits.reshape(-1, 3),
            detect_labels.reshape(-1),
            ignore_index=IGNORE_INDEX,
        )

        # gather gold + negative spans across the batch for the role head
        span_bi: list[tuple[int, int, int]] = []
        role_targets: list[int] = []
        for b, ex_spans in enumerate(spans or []):
            for (s, e, r) in ex_spans:
                span_bi.append((b, s, e))
                role_targets.append(r)

        if span_bi:
            role_logits = self.role_head(self._span_reps(hidden, span_bi))
            targets = torch.tensor(role_targets, device=hidden.device)
            role_loss = F.cross_entropy(role_logits, targets)
        else:
            role_loss = detect_logits.sum() * 0.0  # keep graph; no spans this batch

        loss = detect_loss + self.role_lambda * role_loss
        return {"loss": loss, "detect_logits": detect_logits}
