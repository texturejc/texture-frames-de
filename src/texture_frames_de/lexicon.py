"""
SALSA/German frame lexicon — corpus-free (loads precomputed artifacts).

The training/dev pipeline builds this by parsing the SALSA corpus; the shipped
package instead loads three small JSON artifacts (bundled under `data/`) so it
needs neither the corpus nor NLTK at inference:

  * frame_fe.json      — per-frame core/non-core FE lists + source
  * candidate_map.json — trigger lemma/word -> candidate frames (soft-mask)

German inflection is handled with `simplemma` if available (recommended), else
surface-form matching only. The candidate mask is a soft prior; an empty
candidate list simply makes it a no-op.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from ._tokenization import word_at

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def default_lemmatizer() -> Optional[Callable[[str], str]]:
    try:
        import simplemma

        return lambda w: simplemma.lemmatize(w, lang="de")
    except Exception:
        return None


def _norm(w: str) -> str:
    return w.lower().strip()


class Lexicon:
    def __init__(self, data_dir: str = _DATA):
        with open(os.path.join(data_dir, "frame_fe.json"), encoding="utf-8") as f:
            self._frame_fe = json.load(f)
        with open(os.path.join(data_dir, "candidate_map.json"), encoding="utf-8") as f:
            cm = json.load(f)
        self._lemma_map = cm["lemma"]
        self._word_map = cm["word"]

    # -- per-frame FE access ---------------------------------------------- #
    def frame_elements(self, frame: str) -> tuple[list[str], list[str]]:
        d = self._frame_fe.get(frame)
        return (d["core"], d["non_core"]) if d else ([], [])

    def is_non_core(self, frame: str, fe: str) -> bool:
        d = self._frame_fe.get(frame)
        return bool(d and fe in d["non_core"])

    def frame_source(self, frame: str) -> str:
        d = self._frame_fe.get(frame)
        return d["source"] if d else "?"

    # -- candidate frames (soft-mask) ------------------------------------- #
    def candidate_frames(
        self, text: str, trigger_loc: int,
        lemmatizer: Optional[Callable[[str], str]] = None,
    ) -> list[str]:
        word = word_at(text, trigger_loc)
        out: list[str] = []
        for fr in self._word_map.get(_norm(word), []):
            if fr not in out:
                out.append(fr)
        if lemmatizer is not None:
            try:
                lem = lemmatizer(word)
            except Exception:
                lem = None
            if lem:
                for fr in self._lemma_map.get(_norm(lem), []):
                    if fr not in out:
                        out.append(fr)
        return out
