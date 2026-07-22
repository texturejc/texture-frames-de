"""
Trigger identification for German / SALSA — a lexicon rule, not a learned head.

Why not a neural tagger (as the English parser uses)? SALSA is *partially
annotated*: it selected ~685 target lemmas and annotated their occurrences, so a
sentence's unmarked words are NOT reliable negatives. Measured on the corpus,
83% of covered-lemma token occurrences are unannotated — training a closed-world
token tagger on those as negatives would poison it, and we cannot separate
"genuine non-trigger" from "unannotated trigger".

But SALSA's closed-lemma design turns trigger detection into a *lexicon
membership* problem, and that is strong: build a {lemma -> annotation-rate} table
from the training split, and fire on a token iff its lemma's rate ≥ τ. The rate
threshold alone drops the function-word contaminants (der/zu/… appear as targets
a handful of times but tens of thousands of times overall), so no POS tagger is
needed. Measured (lexicon built on train, evaluated on test, τ=0.5):

    P 0.896 / R 0.926 / F1 0.911     — above the English neural head's 0.750.

Honest about coverage: this only fires on SALSA's ~665 content lemmas (τ=0.5).
That is the real ceiling of a SALSA-trained trigger detector, not a shortcut.

At inference, tokens are lemmatized with simplemma (the same lemmatizer the frame
head's candidate mask uses); pass any callable `lemmatizer(word)->lemma`.
"""
from __future__ import annotations

import collections
import xml.etree.ElementTree as ET
from typing import Callable, Optional

from salsa_loader import DEFAULT_CORPUS, _local, _split_of

DEFAULT_THRESHOLD = 0.5


def _default_lemmatizer() -> Optional[Callable[[str], str]]:
    try:
        import simplemma

        return lambda w: simplemma.lemmatize(w, lang="de")
    except Exception:
        return None


def build_lemma_rate(corpus_path: str = DEFAULT_CORPUS, split: str = "train") -> dict[str, float]:
    """{TIGER-lemma -> annotation rate} over `split`. A lemma's rate is the
    fraction of its token occurrences that are annotated as a frame target."""
    occ: collections.Counter = collections.Counter()
    ann: collections.Counter = collections.Counter()
    for _ev, s in ET.iterparse(corpus_path, events=("end",)):
        if _local(s.tag) != "s":
            continue
        if split is not None and _split_of(s.get("id", "")) != split:
            s.clear()
            continue
        lemma: dict[str, str] = {}
        for c in s:
            if _local(c.tag) != "graph":
                continue
            for sub in c:
                if _local(sub.tag) == "terminals":
                    for t in sub:
                        if _local(t.tag) == "t":
                            lemma[t.get("id")] = t.get("lemma", "")
        targets: set[str] = set()
        for c in s:
            if _local(c.tag) != "sem":
                continue
            for fw in c:
                if _local(fw.tag) != "frames":
                    continue
                for fr in fw:
                    if _local(fr.tag) != "frame":
                        continue
                    for tg in fr:
                        if _local(tg.tag) == "target":
                            for fn in tg:
                                if _local(fn.tag) == "fenode":
                                    targets.add(fn.get("idref"))
        for tid, lem in lemma.items():
            if lem:
                occ[lem] += 1
                ann[lem] += tid in targets
        s.clear()
    return {lem: ann[lem] / occ[lem] for lem in occ if ann[lem] > 0}


# --------------------------------------------------------------------------- #
# whitespace-word helpers (match the loader / frame data conventions)          #
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


class SalsaTriggerDetector:
    """Lexicon-rule trigger detector. Build once (parses the train split to build
    the lemma-rate table), then call `.triggers(text)`."""

    def __init__(
        self,
        threshold: float = DEFAULT_THRESHOLD,
        corpus_path: str = DEFAULT_CORPUS,
        split: str = "train",
        lemma_rate: Optional[dict[str, float]] = None,
        lemmatizer: Optional[Callable[[str], str]] = "default",
    ):
        self.threshold = threshold
        self.lemma_rate = lemma_rate if lemma_rate is not None else build_lemma_rate(corpus_path, split)
        # lemmas that clear the threshold (the "trigger lexicon")
        self.trigger_lemmas = {l for l, r in self.lemma_rate.items() if r >= threshold}
        self.lemmatizer = _default_lemmatizer() if lemmatizer == "default" else lemmatizer

    # -- core lookup ------------------------------------------------------- #
    def _rate_for(self, word: str) -> float:
        """Best annotation rate reachable for a surface word: its simplemma lemma,
        and the lowercased surface form itself (covers forms where the TIGER lemma
        equals the surface, and guards against lemmatizer disagreement)."""
        best = self.lemma_rate.get(word, 0.0)
        best = max(best, self.lemma_rate.get(word.lower(), 0.0))
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
        """Char offsets (word starts) of the trigger words in `text` — the same
        `trigger_loc` convention the frame/args heads consume."""
        return [s for (s, e) in _whitespace_words(text) if self.is_trigger(text[s:e])]


# --------------------------------------------------------------------------- #
# Self-check: honest end-to-end P/R/F1 on the test split                       #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    from salsa_loader import load_trigger_sentences

    det = SalsaTriggerDetector(threshold=DEFAULT_THRESHOLD)
    print(f"trigger lexicon: {len(det.trigger_lemmas)} lemmas (rate ≥ {det.threshold}) "
          f"| lemmatizer: {'simplemma' if det.lemmatizer else 'none (surface only)'}")

    # word-level P/R/F1 against gold on the test split, using the real inference
    # path (reconstructed text + simplemma) — comparable to the English 0.750.
    tp = fp = fn = 0
    for text, gold_locs in load_trigger_sentences("test", drop_unannotated=True):
        words = _whitespace_words(text)
        gold = {i for i, (s, e) in enumerate(words) if any(s <= g < e for g in gold_locs)}
        pred = {i for i, (s, e) in enumerate(words) if det.is_trigger(text[s:e])}
        tp += len(gold & pred); fp += len(pred - gold); fn += len(gold - pred)
    P = tp / (tp + fp) if tp + fp else 0.0
    R = tp / (tp + fn) if tp + fn else 0.0
    F = 2 * P * R / (P + R) if P + R else 0.0
    print(f"\nTEST trigger detection (word-level): P={P:.3f} R={R:.3f} F1={F:.3f}")
    print(f"  tp={tp} fp={fp} fn={fn}")
    print("\nexample:")
    ex = "Die Regierung kündigte an , die Steuern zu erhöhen ."
    word_at = {s: ex[s:e] for (s, e) in _whitespace_words(ex)}
    print(f"  {ex}")
    print(f"  triggers -> {[word_at[loc] for loc in det.triggers(ex)]}")
