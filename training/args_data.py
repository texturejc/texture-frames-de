"""
Argument-extraction (frame-element / role labeling) data + metric.

Frame-conditioned BIO token classification over a global FE-name vocabulary.
Input is a single sequence `"{frame} | {trigger_word} : {sentence}"`; only the
sentence-region tokens carry BIO labels (prefix tokens are ignored). At inference
the label logits are masked to the current frame's FEs so only valid roles are
emitted, and the weighted (FE_name, span_text) metric mirrors upstream
ArgumentsExtractionSample.evaluate_prediction (non-core FEs score 0.5).

Pure functions (no torch/transformers) are unit-tested in tests/.
"""
from __future__ import annotations

from data import (
    TRIGGER_END,
    TRIGGER_START,
    _split_doc_filter,
    snap_to_word_start,
    whitespace_words,
)

IGNORE_INDEX = -100

# Predicate-position markers inserted around the trigger word (M3): tells the
# model *where* the predicate is, not just its identity.
MARK_L = f"{TRIGGER_START} "  # "<t> "
MARK_R = f" {TRIGGER_END}"    # " </t>"


# --------------------------------------------------------------------------- #
# Pure helpers                                                                 #
# --------------------------------------------------------------------------- #

def trigger_word_span(text: str, trigger_loc: int) -> tuple[int, int]:
    """(start, end) of the whitespace word containing trigger_loc."""
    loc = snap_to_word_start(text, trigger_loc)
    for s, e in whitespace_words(text):
        if s <= loc < e:
            return s, e
    return loc, loc


def trigger_word_text(text: str, trigger_loc: int) -> str:
    """The whitespace word containing trigger_loc."""
    s, e = trigger_word_span(text, trigger_loc)
    if e > s:
        return text[s:e]
    tail = text[s:].split()
    return tail[0] if tail else ""


def frame_fe_hint(lexicon, frame: str, max_fes: int = 20) -> str:
    """The frame's FE 'menu' as a string — core roles first, then non-core, capped.
    Fed into the input so the model knows which roles to look for (M3)."""
    core, non_core = lexicon.frame_elements(frame)
    fes = list(core) + list(non_core)
    return "; ".join(fes[:max_fes])


def build_args_input(
    text: str, frame: str, trigger_loc: int, fe_hint: str = ""
) -> tuple[str, int, int, int]:
    """Return (combined_text, prefix_len, ts, te).

    The trigger word is wrapped inline with predicate-position markers so the
    model sees *where* the predicate is (M3), and the frame's FE menu is listed in
    the prefix so it knows *which roles to look for*:
        "{frame} [{FE1}; {FE2}; …] : {…before} <t> {trigger} </t> {after…}"
    ts/te are the trigger word's span in the *original* text — callers pass them
    to `remap_fe_span` to move gold FE offsets through the inserted markers.
    """
    ts, te = trigger_word_span(text, trigger_loc)
    marked = text[:ts] + MARK_L + text[ts:te] + MARK_R + text[te:]
    prefix = f"{frame} [{fe_hint}] : " if fe_hint else f"{frame} : "
    return prefix + marked, len(prefix), ts, te


def remap_fe_span(start: int, end: int, ts: int, te: int, prefix_len: int) -> tuple[int, int]:
    """Move an FE char span from original-sentence coords into the marked
    *combined* coords produced by build_args_input.

    Start boundaries use `>=` and end boundaries use `>` so a span that abuts the
    trigger lands on the correct side of the markers: an FE ending exactly at the
    trigger start is *not* pushed past MARK_L, and an FE starting exactly at the
    trigger start (i.e. containing it) *is*."""
    def shift(p: int, is_end: bool) -> int:
        past_left = (p > ts) if is_end else (p >= ts)
        past_right = (p > te) if is_end else (p >= te)
        return p + (len(MARK_L) if past_left else 0) + (len(MARK_R) if past_right else 0)

    return shift(start, False) + prefix_len, shift(end, True) + prefix_len


def fe_label_maps(fe_vocab: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = ["O"]
    for fe in fe_vocab:
        labels.append(f"B-{fe}")
        labels.append(f"I-{fe}")
    label2id = {lab: i for i, lab in enumerate(labels)}
    id2label = {i: lab for lab, i in label2id.items()}
    return labels, label2id, id2label


def align_fe_bio(
    offset_mapping: list[tuple[int, int]],
    fe_char_spans: list[tuple[int, int, str]],
    label2id: dict[str, int],
    prefix_len: int,
    text: str,
) -> list[int]:
    """Per-token BIO labels. fe_char_spans are (start, end, fe_name) in *combined*
    coords. Prefix/special tokens (end <= prefix_len) get IGNORE_INDEX.

    The token start is snapped past any leading whitespace before the containment
    test: DeBERTa's SentencePiece tokenizer reports a word's token as starting on
    the space before it (the ▁ marker), so a raw `start <= ts < end` test drops
    the FIRST token of every span. Snapping fixes that without over-including the
    token that follows the span (its snapped start lands at/after `end`)."""
    o_id = label2id["O"]
    labels: list[int] = []
    prev_key = None
    for ts, te in offset_mapping:
        if te <= prefix_len:  # special (0,0) or prefix tokens
            labels.append(IGNORE_INDEX)
            prev_key = None
            continue
        ets = snap_to_word_start(text, ts)  # skip the leading-space offset
        found = None
        for s, e, name in fe_char_spans:
            if s <= ets < e:
                found = (s, e, name)
                break
        if found is None:
            labels.append(o_id)
            prev_key = None
        else:
            name = found[2]
            tag = f"I-{name}" if prev_key == found else f"B-{name}"
            labels.append(label2id.get(tag, o_id))
            prev_key = found
    return labels


def decode_bio_spans(
    offset_mapping: list[tuple[int, int]],
    pred_ids: list[int],
    id2label: dict[int, str],
    prefix_len: int,
    combined_text: str,
) -> list[tuple[str, str]]:
    """Decode BIO predictions into [(fe_name, span_text), ...]. Only tokens in the
    sentence region (end > prefix_len) are considered."""
    spans: list[list] = []
    cur = None  # [fe_name, char_start, char_end]
    for (ts, te), pid in zip(offset_mapping, pred_ids):
        if te <= prefix_len:
            if cur:
                spans.append(cur)
                cur = None
            continue
        lab = id2label.get(pid, "O")
        if lab == "O":
            if cur:
                spans.append(cur)
                cur = None
        elif lab.startswith("B-"):
            if cur:
                spans.append(cur)
            cur = [lab[2:], ts, te]
        else:  # I-
            fe = lab[2:]
            if cur and cur[0] == fe:
                cur[2] = te
            else:
                if cur:
                    spans.append(cur)
                cur = [fe, ts, te]
    if cur:
        spans.append(cur)
    return [(fe, _clean_span_text(combined_text[s:e])) for fe, s, e in spans]


def _clean_span_text(t: str) -> str:
    """Drop any predicate markers a span may abut and normalize whitespace."""
    t = t.replace(TRIGGER_START, " ").replace(TRIGGER_END, " ")
    return " ".join(t.split())


def score_args(
    gold: list[tuple[str, str]],
    pred: list[tuple[str, str]],
    is_non_core,
) -> tuple[float, float, float]:
    """Weighted (tp, fp, fn) — mirrors upstream evaluate_prediction. Non-core FEs
    score 0.5, core 1.0. Match requires exact (fe_name, text) tuple equality."""
    def w(fe: str) -> float:
        return 0.5 if is_non_core(fe) else 1.0

    tp = fp = fn = 0.0
    for g in gold:
        if g in pred:
            tp += w(g[0])
        else:
            fn += w(g[0])
    for p in pred:
        if p not in gold:
            fp += w(p[0])
    return tp, fp, fn


# --------------------------------------------------------------------------- #
# FrameNet loading + dataset                                                   #
# --------------------------------------------------------------------------- #

def load_args_examples(split: str) -> list[tuple[str, int, str, list[tuple[str, int, int]]]]:
    """[(text, trigger_loc, frame, [(fe_name, start, end), ...]), ...].

    One example per (annotation, trigger_loc) — mirrors upstream
    ArgumentsExtractionSample generation, incl. the broken-sentence drop.
    """
    import nltk
    from nltk.corpus import framenet as fn

    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        nltk.download("framenet_v17")

    include_docs, exclude_docs = _split_doc_filter(split)

    out = []
    for doc in fn.docs():
        fname = doc["filename"]
        if exclude_docs and fname in exclude_docs:
            continue
        if include_docs and fname not in include_docs:
            continue
        for sentence in doc["sentence"]:
            text = sentence["text"]
            pending = []
            broken = False
            for ann in sentence["annotationSet"]:
                if "FE" in ann and "Target" in ann and "frame" in ann:
                    frame = ann["frame"]["name"]
                    fes = [(fe[2], fe[0], fe[1]) for fe in ann["FE"][0]]
                    for target_span in ann["Target"]:
                        loc = target_span[0]
                        if loc >= len(text):
                            broken = True
                            break
                        pending.append((text, loc, frame, fes))
                if broken:
                    break
            if not broken:
                out.extend(pending)
    return out


def build_args_dataset(split: str, tokenizer, label2id: dict, lexicon, max_length: int = 320):
    """Torch Dataset of input_ids/attention_mask/labels (BIO) for token classification."""
    import torch

    class _ListDataset(torch.utils.data.Dataset):
        def __init__(self, rows):
            self.rows = rows

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, idx):
            return self.rows[idx]

    rows = []
    for text, trigger_loc, frame, fes in load_args_examples(split):
        hint = frame_fe_hint(lexicon, frame)
        combined, prefix_len, ts, te = build_args_input(text, frame, trigger_loc, hint)
        fe_char_spans = [
            (*remap_fe_span(start, end, ts, te, prefix_len), name)
            for name, start, end in fes
        ]
        enc = tokenizer(
            combined, truncation=True, max_length=max_length, return_offsets_mapping=True
        )
        labels = align_fe_bio(
            enc["offset_mapping"], fe_char_spans, label2id, prefix_len, combined
        )
        rows.append(
            {
                "input_ids": enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "labels": labels,
            }
        )
    return _ListDataset(rows)
