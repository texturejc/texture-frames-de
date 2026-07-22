"""
Frame-classification (v2, marker-pooled) data layer for German / SALSA.

The German counterpart of `encoder_parser/frame2_data.py`. The marker-pooling
recipe is language-agnostic — wrap the trigger word in `<t> … </t>` and record
the two marker token indices — so this module differs from the English one only
in its data source: `salsa_loader.load_frame_examples` instead of the NLTK
FrameNet loader. The `<t>`/`</t>` markers and `find_marker_positions` are
identical, kept here so the German pipeline has no dependency on the English
`data.py` (which is FrameNet/NLTK-flavoured).

`Unannotated` (SALSA's "no frame applies" pseudo-frame) is dropped by default —
it is not a frame-evoking label and should not train a frame classifier.
"""
from __future__ import annotations

from salsa_loader import load_frame_examples

# Entity-marker tokens wrapped around the trigger word (same as the English side).
TRIGGER_START = "<t>"
TRIGGER_END = "</t>"


# --------------------------------------------------------------------------- #
# Pure helpers (language-agnostic; mirror data.py / frame2_data.py)            #
# --------------------------------------------------------------------------- #

def _whitespace_words(text: str) -> list[tuple[int, int]]:
    spans, start = [], None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                spans.append((start, i)); start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


def _snap_to_word_start(text: str, idx: int) -> int:
    n = len(text)
    while idx < n and text[idx].isspace():
        idx += 1
    return idx


def mark_trigger(text: str, trigger_loc: int) -> str:
    """Wrap the whitespace word containing `trigger_loc` with `<t> … </t>`.
    With the loader's POS head-anchoring, that word is the target's content head
    (e.g. the finite verb of a separable verb), which is what we want marked."""
    loc = _snap_to_word_start(text, trigger_loc)
    start = end = loc
    for s, e in _whitespace_words(text):
        if s <= loc < e:
            start, end = s, e
            break
    return f"{text[:start]}{TRIGGER_START} {text[start:end]} {TRIGGER_END}{text[end:]}"


def find_marker_positions(input_ids: list[int], start_id: int, end_id: int) -> tuple[int, int]:
    """Token indices of the `<t>` and `</t>` markers. Falls back to CLS (0) / last
    token if a marker was truncated away (rare — trigger near a long tail)."""
    start = input_ids.index(start_id) if start_id in input_ids else 0
    end = input_ids.index(end_id) if end_id in input_ids else len(input_ids) - 1
    return start, end


# --------------------------------------------------------------------------- #
# Dataset builder                                                              #
# --------------------------------------------------------------------------- #

def build_frame2_dataset(
    split: str,
    tokenizer,
    frame2id: dict,
    start_id: int,
    end_id: int,
    max_length: int = 320,
    drop_unannotated: bool = True,
):
    """Torch Dataset: input_ids, attention_mask, start_pos, end_pos, labels(frame id).
    Requires the tokenizer to already have the `<t>`/`</t>` markers added."""
    import torch

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    rows = []
    for text, trigger_loc, frame in load_frame_examples(
        split, drop_unannotated=drop_unannotated
    ):
        if frame not in frame2id:
            continue  # frame outside the label space (shouldn't happen — vocab is a union)
        enc = tokenizer(mark_trigger(text, trigger_loc), truncation=True, max_length=max_length)
        sp, ep = find_marker_positions(enc["input_ids"], start_id, end_id)
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "start_pos": sp,
                "end_pos": ep,
                "labels": frame2id[frame],
            }
        )
    return _ListDataset(rows)
