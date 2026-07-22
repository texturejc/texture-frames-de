"""
Data preparation for the encoder trigger-identification head.

Split into two layers:
  * pure-Python helpers (word segmentation, trigger-word id, label alignment,
    word-level scoring) — no torch/transformers, unit-tested in tests/.
  * a builder that uses the upstream FrameNet loaders + a HF fast tokenizer to
    produce a tokenized dataset for `AutoModelForTokenClassification`.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Iterable, Optional

# protobuf C++ backend guard (see requirements-colab.txt / notebooks)
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

# Make the vendored parser importable from source (no install needed).
_REPO = Path(__file__).resolve().parent.parent / "frame-semantic-transformer"
if _REPO.exists() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# Label scheme for the token-classification head.
LABELS = ["O", "TRIGGER"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}
IGNORE_INDEX = -100  # torch CrossEntropyLoss / HF default ignore id

# Entity-marker tokens wrapped around the trigger word for frame classification.
TRIGGER_START = "<t>"
TRIGGER_END = "</t>"


# --------------------------------------------------------------------------- #
# Pure-Python core (no torch/transformers) — unit-tested                       #
# --------------------------------------------------------------------------- #

def whitespace_words(text: str) -> list[tuple[int, int]]:
    """Return (start, end) char spans of maximal non-whitespace runs.

    Matches the whitespace `.split()` word definition used by the upstream
    trigger metric, but keeps char offsets so we can align to trigger_locs.
    """
    spans: list[tuple[int, int]] = []
    start: Optional[int] = None
    for i, ch in enumerate(text):
        if ch.isspace():
            if start is not None:
                spans.append((start, i))
                start = None
        elif start is None:
            start = i
    if start is not None:
        spans.append((start, len(text)))
    return spans


def snap_to_word_start(text: str, idx: int) -> int:
    """Advance `idx` to the first non-whitespace char at or after it.

    DeBERTa-v3's SentencePiece fast tokenizer can report a sub-token offset that
    starts on the leading space marker rather than the first letter; snapping
    makes trigger-word alignment robust to that quirk. A no-op for clean offsets.
    """
    n = len(text)
    while idx < n and text[idx].isspace():
        idx += 1
    return idx


def trigger_word_indices(
    words: list[tuple[int, int]], trigger_locs: Iterable[int]
) -> set[int]:
    """Indices of `words` that are frame triggers.

    A word is a trigger iff some trigger_loc falls within its char span. Upstream
    inserts a `*` at exactly the trigger_loc, which is the start of the trigger
    word, so this coincides with word-start matching while tolerating minor
    off-by-one offsets.
    """
    locs = sorted(set(trigger_locs))
    out: set[int] = set()
    for idx, (start, end) in enumerate(words):
        if any(start <= loc < end for loc in locs):
            out.add(idx)
    return out


def align_trigger_labels(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[Optional[int]],
    word_is_trigger: list[bool],
) -> list[int]:
    """Per-token labels for token classification.

    Standard first-subword scheme: the first sub-token of each word carries the
    word's label; continuation sub-tokens and special tokens get IGNORE_INDEX so
    they don't contribute to the loss or to per-word scoring.

    `offset_mapping` is accepted for symmetry/validation; grouping uses word_ids.
    """
    assert len(offset_mapping) == len(word_ids)
    labels: list[int] = []
    prev_word: Optional[int] = None
    for wid in word_ids:
        if wid is None:
            labels.append(IGNORE_INDEX)
        elif wid != prev_word:
            labels.append(LABEL2ID["TRIGGER"] if word_is_trigger[wid] else LABEL2ID["O"])
        else:
            labels.append(IGNORE_INDEX)
        prev_word = wid
    return labels


def predicted_trigger_locs_from_tokens(
    offset_mapping: list[tuple[int, int]],
    word_ids: list[Optional[int]],
    token_pred_is_trigger: list[bool],
) -> set[int]:
    """Map first-subword token predictions back to word-start char offsets."""
    locs: set[int] = set()
    prev_word: Optional[int] = None
    for (start, _end), wid, is_trig in zip(
        offset_mapping, word_ids, token_pred_is_trigger
    ):
        if wid is not None and wid != prev_word and is_trig:
            locs.add(start)
        prev_word = wid
    return locs


def score_trigger_words(
    text: str,
    gold_trigger_locs: Iterable[int],
    pred_trigger_locs: Iterable[int],
) -> tuple[int, int, int]:
    """Word-level (true_pos, false_pos, false_neg), upstream-comparable.

    A word counts as a gold/predicted trigger iff a gold/predicted loc lands in
    its whitespace span.
    """
    words = whitespace_words(text)
    gold = trigger_word_indices(words, (snap_to_word_start(text, loc) for loc in gold_trigger_locs))
    pred = trigger_word_indices(words, (snap_to_word_start(text, loc) for loc in pred_trigger_locs))
    true_pos = len(gold & pred)
    false_pos = len(pred - gold)
    false_neg = len(gold - pred)
    return true_pos, false_pos, false_neg


def mark_trigger(text: str, trigger_loc: int) -> str:
    """Wrap the trigger word (the whitespace word containing trigger_loc) with
    entity-marker tokens, e.g. 'The chef <t> gave </t> food.' — the input format
    for the frame-classification head. Pure/testable."""
    loc = snap_to_word_start(text, trigger_loc)
    start = end = loc
    for s, e in whitespace_words(text):
        if s <= loc < e:
            start, end = s, e
            break
    return f"{text[:start]}{TRIGGER_START} {text[start:end]} {TRIGGER_END}{text[end:]}"


def prf1(true_pos: int, false_pos: int, false_neg: int) -> dict[str, float]:
    precision = true_pos / (true_pos + false_pos) if (true_pos + false_pos) else 0.0
    recall = true_pos / (true_pos + false_neg) if (true_pos + false_neg) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


# --------------------------------------------------------------------------- #
# FrameNet loading + tokenization (needs nltk + transformers)                  #
# --------------------------------------------------------------------------- #

def load_trigger_sentences(split: str) -> list[tuple[str, list[int]]]:
    """Return [(sentence_text, [trigger_locs]), ...] for a split.

    split ∈ {"train", "dev", "test"}.

    Fully decoupled from `frame_semantic_transformer`: importing ANY submodule
    under its `framenet17` package runs that package's __init__, which imports
    Framenet17TrainingLoader -> the augmentation classes (SynonymAugmentation,
    KeyboardAugmentation) -> `nlpaug`. We never run augmentation, so we vendor the
    split lists (`sesame_splits.py`) and read the corpus via nltk directly. The
    parsing below mirrors upstream
    `parse_annotated_sentence_from_framenet_sentence` for the trigger fields, so
    the sentence set + trigger locs match what the baseline was scored on.
    """
    import nltk
    from nltk.corpus import framenet as fn

    from sesame_splits import SESAME_DEV_FILES, SESAME_TEST_FILES

    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        nltk.download("framenet_v17")

    if split == "train":
        include_docs, exclude_docs = None, set(SESAME_DEV_FILES) | set(SESAME_TEST_FILES)
    elif split == "dev":
        include_docs, exclude_docs = set(SESAME_DEV_FILES), None
    elif split == "test":
        include_docs, exclude_docs = set(SESAME_TEST_FILES), None
    else:
        raise ValueError(f"unknown split {split!r}")

    out: list[tuple[str, list[int]]] = []
    for doc in fn.docs():
        fname = doc["filename"]
        if exclude_docs and fname in exclude_docs:
            continue
        if include_docs and fname not in include_docs:
            continue
        for sentence in doc["sentence"]:
            text = sentence["text"]
            locs: list[int] = []
            broken = False
            for ann in sentence["annotationSet"]:
                if "FE" in ann and "Target" in ann and "frame" in ann:
                    for target_span in ann["Target"]:
                        loc = target_span[0]
                        if loc >= len(text):
                            broken = True  # upstream drops the whole sentence
                            break
                        locs.append(loc)
                if broken:
                    break
            # upstream keeps the sentence only if it has ≥1 valid frame annotation
            if not broken and locs:
                out.append((text, sorted(set(locs))))
    return out


def build_trigger_dataset(split: str, tokenizer, max_length: int = 320):
    """Tokenize a split into a torch Dataset of input_ids/attention_mask/labels.

    Uses a plain `torch.utils.data.Dataset`, NOT huggingface `datasets` — the
    latter pulls in pyarrow, whose compiled extensions are a recurring source of
    numpy-ABI crashes on Colab. HF `Trainer` accepts any map-style torch Dataset.
    Requires a *fast* tokenizer (offset_mapping + word_ids).
    """
    import torch

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    rows = []
    for text, trigger_locs in load_trigger_sentences(split):
        enc = tokenizer(
            text, truncation=True, max_length=max_length, return_offsets_mapping=True
        )
        word_ids = enc.word_ids()
        words = whitespace_words(text)
        gold = trigger_word_indices(words, trigger_locs)

        # Per-tokenizer-word trigger flag: a tokenizer "word" is a trigger iff its
        # (snapped) char-span start falls in a gold trigger whitespace-word.
        n_words = max((w for w in word_ids if w is not None), default=-1) + 1
        word_is_trigger = [False] * n_words
        seen = [False] * n_words
        for (s, _e), wid in zip(enc["offset_mapping"], word_ids):
            if wid is None or seen[wid]:
                continue
            seen[wid] = True
            cs = snap_to_word_start(text, s)
            word_is_trigger[wid] = any(words[gi][0] <= cs < words[gi][1] for gi in gold)

        labels = align_trigger_labels(enc["offset_mapping"], word_ids, word_is_trigger)
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }
        )
    return _ListDataset(rows)


# --------------------------------------------------------------------------- #
# Frame classification (slice 2)                                               #
# --------------------------------------------------------------------------- #

def _split_doc_filter(split: str):
    from sesame_splits import SESAME_DEV_FILES, SESAME_TEST_FILES

    if split == "train":
        return None, set(SESAME_DEV_FILES) | set(SESAME_TEST_FILES)
    if split == "dev":
        return set(SESAME_DEV_FILES), None
    if split == "test":
        return set(SESAME_TEST_FILES), None
    raise ValueError(f"unknown split {split!r}")


def load_frame_examples(split: str) -> list[tuple[str, int, str]]:
    """Return [(sentence_text, trigger_loc, gold_frame_name), ...] for a split.

    One example per (annotation, trigger_loc) — mirrors upstream
    tasks_from_annotated_sentences' FrameClassificationSample generation, incl.
    the "drop the whole sentence if any trigger loc is out of range" rule.
    """
    import nltk
    from nltk.corpus import framenet as fn

    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        nltk.download("framenet_v17")

    include_docs, exclude_docs = _split_doc_filter(split)

    out: list[tuple[str, int, str]] = []
    for doc in fn.docs():
        fname = doc["filename"]
        if exclude_docs and fname in exclude_docs:
            continue
        if include_docs and fname not in include_docs:
            continue
        for sentence in doc["sentence"]:
            text = sentence["text"]
            pending: list[tuple[str, int, str]] = []
            broken = False
            for ann in sentence["annotationSet"]:
                if "FE" in ann and "Target" in ann and "frame" in ann:
                    frame = ann["frame"]["name"]
                    for target_span in ann["Target"]:
                        loc = target_span[0]
                        if loc >= len(text):
                            broken = True
                            break
                        pending.append((text, loc, frame))
                if broken:
                    break
            if not broken:
                out.extend(pending)
    return out


def build_frame_dataset(split: str, tokenizer, frame2id: dict, max_length: int = 320):
    """Tokenize marked-trigger sentences into a torch Dataset for sequence
    classification. Label = frame id. Requires the tokenizer to already have the
    entity-marker tokens added (train_frame.py does this)."""
    import torch

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    rows = []
    for text, trigger_loc, frame in load_frame_examples(split):
        if frame not in frame2id:
            continue  # frame absent from the FrameNet vocab (shouldn't happen)
        enc = tokenizer(
            mark_trigger(text, trigger_loc), truncation=True, max_length=max_length
        )
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": frame2id[frame],
            }
        )
    return _ListDataset(rows)
