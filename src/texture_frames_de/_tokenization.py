"""
Pure tokenization / trigger-marking helpers (no torch/transformers).

Language-agnostic; shared by the frame and argument stages. Vendored so the
package is self-contained.
"""
from __future__ import annotations

from typing import Optional

TRIGGER_START = "<t>"
TRIGGER_END = "</t>"


def whitespace_words(text: str) -> list[tuple[int, int]]:
    """(start, end) char spans of maximal non-whitespace runs."""
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
    """Advance idx to the first non-whitespace char at or after it."""
    n = len(text)
    while idx < n and text[idx].isspace():
        idx += 1
    return idx


def word_at(text: str, loc: int) -> str:
    """The whitespace word containing (or starting at/after) loc."""
    loc = snap_to_word_start(text, loc)
    for s, e in whitespace_words(text):
        if s <= loc < e:
            return text[s:e]
    return ""


def mark_trigger(text: str, trigger_loc: int) -> str:
    """Wrap the whitespace word containing trigger_loc with `<t> … </t>`."""
    loc = snap_to_word_start(text, trigger_loc)
    start = end = loc
    for s, e in whitespace_words(text):
        if s <= loc < e:
            start, end = s, e
            break
    return f"{text[:start]}{TRIGGER_START} {text[start:end]} {TRIGGER_END}{text[end:]}"


def find_marker_positions(input_ids: list[int], start_id: int, end_id: int) -> tuple[int, int]:
    """Token indices of the `<t>` and `</t>` markers (fallback to CLS/last)."""
    start = input_ids.index(start_id) if start_id in input_ids else 0
    end = input_ids.index(end_id) if end_id in input_ids else len(input_ids) - 1
    return start, end
