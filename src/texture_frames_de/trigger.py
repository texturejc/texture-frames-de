"""
Trigger identification — the SALSA lexicon rule (corpus-free).

Not a neural head: SALSA is partially annotated, so a closed-world token tagger
would be unsound. Instead a token fires as a trigger iff its lemma's annotation
rate (fraction of its corpus occurrences that were annotated as a frame target)
is >= a threshold. Built on train, this scores F1 ~0.88 on test — above a neural
tagger — and is honest about coverage: it fires only on SALSA's ~665 covered
content lemmas at threshold 0.5.

The lemma-rate table (`trigger_rates.json`) is precomputed and bundled; German
inflection is resolved with `simplemma` if available.
"""
from __future__ import annotations

import json
import os
from typing import Callable, Optional

from ._tokenization import whitespace_words
from .lexicon import default_lemmatizer

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
DEFAULT_THRESHOLD = 0.5


class TriggerDetector:
    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        data_dir: str = _DATA,
        lemmatizer: Optional[Callable[[str], str]] = "default",
    ):
        self.threshold = threshold
        with open(os.path.join(data_dir, "trigger_rates.json"), encoding="utf-8") as f:
            self.lemma_rate: dict[str, float] = json.load(f)
        self.lemmatizer = default_lemmatizer() if lemmatizer == "default" else lemmatizer

    def _rate_for(self, word: str) -> float:
        best = max(self.lemma_rate.get(word, 0.0), self.lemma_rate.get(word.lower(), 0.0))
        if self.lemmatizer is not None:
            try:
                lem = self.lemmatizer(word)
            except Exception:
                lem = None
            if lem:
                best = max(best, self.lemma_rate.get(lem, 0.0))
        return best

    def is_trigger(self, word: str) -> bool:
        return self._rate_for(word) >= self.threshold

    def triggers(self, text: str) -> list[int]:
        """Char offsets (word starts) of the trigger words in text."""
        return [s for (s, e) in whitespace_words(text) if self.is_trigger(text[s:e])]


# --------------------------------------------------------------------------- #
# Open-vocabulary trigger detection (optional; needs HanTa for German POS)     #
# --------------------------------------------------------------------------- #

_HANTA = None


def _hanta():
    """Lazily load the HanTa German tagger (optional dependency)."""
    global _HANTA
    if _HANTA is None:
        try:
            from HanTa import HanoverTagger

            _HANTA = HanoverTagger.HanoverTagger("morphmodel_ger.pgz")
        except Exception as e:  # pragma: no cover
            raise ImportError(
                "open-vocabulary mode needs HanTa — `pip install HanTa` "
                "(or install the package with the [open-vocab] extra)."
            ) from e
    return _HANTA


def content_verb_locs(text: str) -> list[int]:
    """Char offsets of full-verb tokens (STTS `VV*`), via HanTa POS tagging.

    These are the reliable frame-evokers the SALSA lexicon rule misses on
    out-of-domain text (e.g. everyday verbs like `essen`). Nouns are deliberately
    excluded here — they are far noisier as triggers.
    """
    words = whitespace_words(text)
    if not words:
        return []
    tags = _hanta().tag_sent([text[s:e] for (s, e) in words])
    out = []
    for (s, _e), (_w, _lem, pos) in zip(words, tags):
        if pos.startswith("VV"):  # VV(FIN)/VV(INF)/VV(PP)/VV(IZU)
            out.append(s)
    return out
