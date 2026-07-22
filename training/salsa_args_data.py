"""
Argument-extraction (v2, detect-then-classify) data layer for German / SALSA.

The German counterpart of `encoder_parser/args2_data.py`. The detect-then-classify
machinery is entirely language-agnostic — predicate-marked, FE-menu-conditioned
input; a 3-class BIO detection target; per-span (token-range, role-id) targets;
sampled NULL negatives — so this module *reuses those pure helpers unchanged*
from the English tree and only swaps the data source:

    salsa_loader.load_args_examples   (was the NLTK FrameNet loader)

Two SALSA-specific choices:

  * `drop_unannotated=True` — the `Unannotated` pseudo-frame is not a real frame
    and is excluded (as in the frame head).
  * `keep_discontinuous` — 13.8% of German gold role spans have a non-contiguous
    constituent yield; the loader represents those as an enclosing char span
    (min-start..max-end), which over-covers. Keeping them (default) preserves all
    role supervision at the cost of some boundary noise; dropping them trains only
    on clean contiguous spans. This is the knob for that experiment.

No German synonym augmentation yet — the English path uses WordNet, which has no
drop-in German equivalent here (GermaNet is license-gated); left for later.
"""
from __future__ import annotations

import os
import sys

# Reuse the language-agnostic pure helpers + label maps from the English tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from args_data import (  # noqa: E402
    IGNORE_INDEX,
    build_args_input,
    frame_fe_hint,
    remap_fe_span,
)
from args2_data import (  # noqa: E402  (re-exported for the trainer/eval)
    DETECT_LABELS,
    NULL_ROLE,
    decode_detect_spans,
    detect_bio_labels,
    gold_span_token_indices,
    role_label_maps,
    sample_negative_spans,
)

from salsa_loader import load_args_examples  # noqa: E402

__all__ = [
    "NULL_ROLE", "DETECT_LABELS", "IGNORE_INDEX",
    "role_label_maps", "decode_detect_spans",
    "build_args_input", "frame_fe_hint", "remap_fe_span",
    "build_args2_dataset",
]


def build_args2_dataset(
    split: str,
    tokenizer,
    role2id: dict,
    lexicon,
    max_length: int = 320,
    n_negatives: int = 4,
    keep_discontinuous: bool = True,
):
    """Torch Dataset of rows: input_ids, attention_mask, detect_labels (3-class BIO),
    and `spans` = [(start_tok, end_tok_inclusive, role_id), ...] — gold FE spans plus
    `n_negatives` sampled NULL spans per example so the role head learns to reject
    spurious detections. Deterministic negative sampling (seeded per example)."""
    import random

    import torch

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    examples = list(
        load_args_examples(
            split, keep_discontinuous=keep_discontinuous, drop_unannotated=True
        )
    )

    null_id = role2id[NULL_ROLE]
    rows = []
    for i, (text, trigger_loc, frame, fes) in enumerate(examples):
        hint = frame_fe_hint(lexicon, frame)
        combined, prefix_len, ts, te = build_args_input(text, frame, trigger_loc, hint)
        remapped = [
            (*remap_fe_span(s, e, ts, te, prefix_len), name) for name, s, e in fes
        ]
        enc = tokenizer(
            combined, truncation=True, max_length=max_length, return_offsets_mapping=True
        )
        n_tok = len(enc["input_ids"])
        om = enc["offset_mapping"]
        detect = detect_bio_labels(om, [(s, e) for s, e, _ in remapped], prefix_len, combined)
        gold = [
            (a, b, name)
            for (a, b, name) in gold_span_token_indices(om, remapped, prefix_len, combined)
            if name in role2id and b < n_tok
        ]
        span_records = [(a, b, role2id[name]) for (a, b, name) in gold]

        sent_toks = [j for j, (s, e) in enumerate(om) if e > prefix_len]
        gold_ranges = {(a, b) for (a, b, _) in gold}
        rng = random.Random(1234 + i)  # deterministic per example
        for (a, b) in sample_negative_spans(sent_toks, gold_ranges, n_negatives, rng):
            span_records.append((a, b, null_id))

        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "detect_labels": detect,
                "spans": span_records,
            }
        )
    return _ListDataset(rows)
