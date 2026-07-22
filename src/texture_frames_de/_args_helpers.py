"""
Pure argument-extraction helpers (no torch/transformers), vendored.

Builds the predicate-marked, FE-menu-conditioned input the args model expects,
and decodes the detection head's BIO output into spans.
"""
from __future__ import annotations

from ._tokenization import TRIGGER_END, TRIGGER_START, snap_to_word_start, whitespace_words

NULL_ROLE = "<NULL>"

# Predicate-position markers inserted around the trigger word.
MARK_L = f"{TRIGGER_START} "  # "<t> "
MARK_R = f" {TRIGGER_END}"    # " </t>"

# 3-class span-detection scheme (role-agnostic).
DETECT_O, DETECT_B, DETECT_I = 0, 1, 2


def trigger_word_span(text: str, trigger_loc: int) -> tuple[int, int]:
    loc = snap_to_word_start(text, trigger_loc)
    for s, e in whitespace_words(text):
        if s <= loc < e:
            return s, e
    return loc, loc


def frame_fe_hint(lexicon, frame: str, max_fes: int = 20) -> str:
    """The frame's FE 'menu' — core roles first, then non-core, capped."""
    core, non_core = lexicon.frame_elements(frame)
    fes = list(core) + list(non_core)
    return "; ".join(fes[:max_fes])


def build_args_input(text: str, frame: str, trigger_loc: int, fe_hint: str = "") -> tuple[str, int, int, int]:
    """(combined_text, prefix_len, ts, te). Wraps the trigger inline with
    predicate markers and lists the FE menu in the prefix."""
    ts, te = trigger_word_span(text, trigger_loc)
    marked = text[:ts] + MARK_L + text[ts:te] + MARK_R + text[te:]
    prefix = f"{frame} [{fe_hint}] : " if fe_hint else f"{frame} : "
    return prefix + marked, len(prefix), ts, te


def decode_detect_spans(offset_mapping, detect_pred, prefix_len):
    """Decode 3-class BIO predictions into
    (first_token, last_token_inclusive, char_start, char_end) spans."""
    spans: list[list[int]] = []
    cur = None
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
    return [tuple(s) for s in spans]


def clean_span_text(t: str) -> str:
    """Drop any predicate markers a span may abut and normalize whitespace."""
    t = t.replace(TRIGGER_START, " ").replace(TRIGGER_END, " ")
    return " ".join(t.split())
