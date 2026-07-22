"""
Argument extraction v2 — data layer for the *detect-then-classify* span head.

See ARGS_V2_DESIGN.md. Two supervision signals per example, both derived from the
same candidate-conditioned, predicate-marked input as v1
(`{frame} [{FE menu}] : … <t> {trigger} </t> …`, built by args_data):

  * Head A — span *detection*: a 3-class BIO label (O / B / I) per token, role-
    agnostic ("is this token part of *an* argument"). Dense signal, arbitrary
    length.
  * Head B — *role* classification: for each gold span, the token range it covers
    plus its FE-name id, so the role head can be trained on pooled span reps.

All functions here are pure (no torch/transformers) and unit-tested, reusing the
offset/marker machinery from args_data so v1 and v2 stay consistent.
"""
from __future__ import annotations

from args_data import (
    IGNORE_INDEX,
    build_args_input,
    frame_fe_hint,
    load_args_examples,
    remap_fe_span,
)
from data import snap_to_word_start

# 3-class span-detection scheme (role-agnostic).
DETECT_LABELS = ["O", "B", "I"]
DETECT_O, DETECT_B, DETECT_I = 0, 1, 2
NULL_ROLE = "<NULL>"


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #

def role_label_maps(fe_vocab: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """Span-level role labels: NULL (reject) + one class per FE. Much smaller /
    better-conditioned than v1's ~2,400-way per-token BIO, and applied once per
    span, not per token."""
    roles = [NULL_ROLE] + list(fe_vocab)
    role2id = {r: i for i, r in enumerate(roles)}
    id2role = {i: r for r, i in role2id.items()}
    return roles, role2id, id2role


def detect_bio_labels(
    offset_mapping: list[tuple[int, int]],
    span_char_ranges: list[tuple[int, int]],
    prefix_len: int,
    text: str,
) -> list[int]:
    """Per-token 3-class BIO detection labels. span_char_ranges are (start, end) in
    *combined* coords (role-agnostic union of the gold FE spans). Prefix/special
    tokens (end <= prefix_len) get IGNORE_INDEX. Token start is snapped past the
    DeBERTa leading-space offset, as in args_data.align_fe_bio."""
    labels: list[int] = []
    prev_span = None
    for ts, te in offset_mapping:
        if te <= prefix_len:
            labels.append(IGNORE_INDEX)
            prev_span = None
            continue
        ets = snap_to_word_start(text, ts)
        found = None
        for s, e in span_char_ranges:
            if s <= ets < e:
                found = (s, e)
                break
        if found is None:
            labels.append(DETECT_O)
            prev_span = None
        else:
            labels.append(DETECT_I if prev_span == found else DETECT_B)
            prev_span = found
    return labels


def gold_span_token_indices(
    offset_mapping: list[tuple[int, int]],
    fe_char_spans: list[tuple[int, int, str]],
    prefix_len: int,
    text: str,
) -> list[tuple[int, int, str]]:
    """For each gold FE span, the (first_token, last_token_inclusive, fe_name) it
    covers — the training targets for the role head. Spans whose tokens all fall in
    the prefix or get truncated away yield no entry."""
    out: list[tuple[int, int, str]] = []
    for s, e, name in fe_char_spans:
        toks = [
            i
            for i, (ts, te) in enumerate(offset_mapping)
            if te > prefix_len and s <= snap_to_word_start(text, ts) < e
        ]
        if toks:
            out.append((toks[0], toks[-1], name))
    return out


def sample_negative_spans(
    sent_tok_indices: list[int],
    gold_ranges: set[tuple[int, int]],
    k: int,
    rng,
    max_width: int = 5,
) -> list[tuple[int, int]]:
    """Sample up to k contiguous token ranges from the sentence region that are NOT
    gold spans — labeled NULL so the role head learns to reject Head-A's spurious
    detections (v1's #1 false-positive source). `rng` is a seeded random.Random for
    determinism; returns (start_tok, end_tok_inclusive) pairs."""
    if not sent_tok_indices:
        return []
    lo, hi = sent_tok_indices[0], sent_tok_indices[-1]
    out: list[tuple[int, int]] = []
    seen = set(gold_ranges)
    tries = 0
    while len(out) < k and tries < k * 10 + 10:
        tries += 1
        s = rng.randint(lo, hi)
        e = s + rng.randint(0, min(max_width - 1, hi - s))
        if (s, e) in seen:
            continue
        seen.add((s, e))
        out.append((s, e))
    return out


def decode_detect_spans(
    offset_mapping: list[tuple[int, int]],
    detect_pred: list[int],
    prefix_len: int,
) -> list[tuple[int, int, int, int]]:
    """Decode 3-class BIO predictions into detected spans as
    (first_token, last_token_inclusive, char_start, char_end). A stray I without a
    B opens a new span (robust decoding). Prefix/special tokens close the current
    span."""
    spans: list[list[int]] = []
    cur: list[int] | None = None
    for i, ((ts, te), p) in enumerate(zip(offset_mapping, detect_pred)):
        if te <= prefix_len:
            if cur:
                spans.append(cur)
                cur = None
            continue
        if p == DETECT_B:
            if cur:
                spans.append(cur)
            cur = [i, i, ts, te]
        elif p == DETECT_I:
            if cur:
                cur[1] = i
                cur[3] = te
            else:
                cur = [i, i, ts, te]
        else:  # O
            if cur:
                spans.append(cur)
                cur = None
    if cur:
        spans.append(cur)
    return [tuple(s) for s in spans]  # type: ignore[misc]


# --------------------------------------------------------------------------- #
# Dataset                                                                      #
# --------------------------------------------------------------------------- #

def build_args2_dataset(
    split: str, tokenizer, role2id: dict, lexicon, max_length: int = 320,
    n_negatives: int = 4, augment: int = 0,
):
    """Torch Dataset of rows: input_ids, attention_mask, detect_labels (BIO 3-class),
    and `spans` = [(start_tok, end_tok_inclusive, role_id), ...] for the role head —
    gold spans (real FE role) plus `n_negatives` sampled NULL spans per example so
    the role head learns to reject spurious detections.

    `augment` (train split only): number of synonym-augmented copies to add per
    arg-bearing example — recovers the long-tail edge the baseline gets from nlpaug."""
    import random

    import torch

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    examples = list(load_args_examples(split))
    if split == "train" and augment > 0:
        from augment import augment_example, wordnet_synonym

        arng = random.Random(2024)
        extra = []
        for (text, loc, frame, fes) in examples:
            if not fes:
                continue  # only augment examples that carry arguments
            for _ in range(augment):
                res = augment_example(text, loc, fes, wordnet_synonym, arng)
                if res:
                    extra.append((res[0], res[1], frame, res[2]))
        print(f"augmentation: +{len(extra)} examples (x{augment} on arg-bearing)")
        examples += extra

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
