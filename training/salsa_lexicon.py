"""
SALSA / German frame lexicon — the counterpart of `texture_frames.lexicon`.

Provides the same surface the English pipeline consumes, sourced from SALSA
instead of NLTK FrameNet:

    frames()                      -> [{name, lexical_units, core_elements,
                                       non_core_elements}, ...]
    frame_vocab() / frame2id()    -> the frame-classification label space
    fe_vocab()                    -> sorted unique role names (BIO role space)
    frame_elements(frame)         -> (core_elements, non_core_elements)
    is_non_core(frame, fe)        -> bool   (SALSA/Sesame scoring: non-core = 0.5)
    candidate_frames(text, loc)   -> [frame, ...]  (soft-mask candidates)

Two SALSA-specific realities shape this module:

  * `salsa_frames.xml` is NOT strict XML (unescaped `&` in example prose), so it
    is read leniently by regex, block by block — never with a validating parser.

  * The dictionary's lexical units are sparse (~1.8 LU/frame; SALSA only records
    the LUs it annotated), and — unlike the English side — we have no rich LU
    inventory to normalise. So the trigger->candidate-frame map is built from
    BOTH the dictionary LUs AND the empirical **train-split** lemma/word→frame
    co-occurrences harvested from the corpus. Candidate lookup at inference is
    therefore surface/lemma based; pass a German lemmatizer to `candidate_frames`
    for inflection-robust recall (dependency-free default matches on the
    lowercased surface word, which covers the forms seen in training).

Pure stdlib. The empirical harvest imports `salsa_loader` (also stdlib).
"""
from __future__ import annotations

import os
import re
from collections import defaultdict
from functools import lru_cache
from typing import Callable, Iterable, Optional

from salsa_loader import DEFAULT_CORPUS, DEFAULT_FRAMES, iter_frame_instances

# FE coreType casing in SALSA is inconsistent; normalise to core vs non-core.
# FrameNet/Sesame scoring counts non-core (peripheral / extra-thematic) at 0.5.
_CORE_TYPES = {"core", "core-unexpressed"}   # lowercased


def _strip_lu_pos(lu_name: str) -> str:
    """'Abnehmer.n' / 'ablehnen.v' -> lemma without the trailing .pos suffix."""
    return re.sub(r"\.[a-zA-Z]+$", "", lu_name).strip()


def _norm_word(w: str) -> str:
    """Lookup key for a trigger surface form / lemma: lowercased, trimmed."""
    return w.lower().strip()


class SalsaLexicon:
    """Frame + FE inventory from salsa_frames.xml, plus a trigger→candidate-frame
    map from the dictionary and the training corpus. Build once; cached."""

    def __init__(
        self,
        frames_path: str = DEFAULT_FRAMES,
        corpus_path: str = DEFAULT_CORPUS,
        candidate_splits: Iterable[str] = ("train",),
    ):
        self.frames_path = frames_path
        self.corpus_path = corpus_path
        # which corpus splits to harvest lemma/word->frame from; train-only by
        # default so evaluation on dev/test sees no candidate-set leakage.
        self.candidate_splits = tuple(candidate_splits)

    # -- frame dictionary (salsa_frames.xml) ------------------------------- #

    @lru_cache(1)
    def frames(self) -> list[dict]:
        """[{name, source, lexical_units, core_elements, non_core_elements}, ...].

        Parsed leniently: split the file into `<frame ...> ... </frame>` blocks
        and regex the fields within each (the file is not well-formed XML)."""
        text = open(self.frames_path, encoding="utf-8", errors="replace").read()
        out: list[dict] = []
        for block in re.findall(r"<frame\b.*?</frame>", text, re.S):
            head = re.match(r"<frame\b([^>]*)>", block)
            attrs = head.group(1) if head else ""
            nm = re.search(r"\bname=['\"]([^'\"]+)", attrs)
            if not nm:
                continue
            name = nm.group(1)
            src = re.search(r"\bsource=['\"]([^'\"]+)", attrs)

            core, non_core = [], []
            for fe in re.finditer(r"<fe\b([^>]*)>", block):
                fa = fe.group(1)
                fn = re.search(r"\bname=['\"]([^'\"]+)", fa)
                ct = re.search(r"\bcoreType=['\"]([^'\"]+)", fa)
                if not fn:
                    continue
                is_core = (ct.group(1).lower() in _CORE_TYPES) if ct else False
                (core if is_core else non_core).append(fn.group(1))

            lus = [
                _strip_lu_pos(m.group(1))
                for m in re.finditer(r"<lexunit\b[^>]*\bname=['\"]([^'\"]+)", block)
            ]

            out.append({
                "name": name,
                "source": src.group(1) if src else "?",
                "lexical_units": lus,
                "core_elements": core,
                "non_core_elements": non_core,
            })
        return out

    @lru_cache(1)
    def _by_name(self) -> dict[str, dict]:
        return {f["name"]: f for f in self.frames()}

    # -- vocabularies ------------------------------------------------------ #

    @lru_cache(1)
    def frame_vocab(self) -> list[str]:
        """Sorted frame names — the classification label space. Union of frames
        DEFINED in the dictionary and frames USED in the corpus (the corpus uses
        a few frames absent from the dictionary; both must be labelable)."""
        names = {f["name"] for f in self.frames()}
        names |= {fi.frame for fi in iter_frame_instances(self.corpus_path)}
        return sorted(names)

    @lru_cache(1)
    def frame2id(self) -> dict[str, int]:
        return {name: i for i, name in enumerate(self.frame_vocab())}

    @lru_cache(1)
    def fe_vocab(self) -> list[str]:
        """Sorted unique FE names — the BIO role space. Union of dictionary FEs
        and roles actually annotated in the corpus (covers roles used but not in
        a frame's dictionary FE list, plus SALSA's added 'Beneficient')."""
        names: set[str] = set()
        for f in self.frames():
            names.update(f["core_elements"])
            names.update(f["non_core_elements"])
        for fi in iter_frame_instances(self.corpus_path):
            names.update(r.role for r in fi.roles)
        return sorted(names)

    # -- per-frame FE access ----------------------------------------------- #

    def frame_elements(self, frame_name: str) -> tuple[list[str], list[str]]:
        """(core_elements, non_core_elements) for a frame ([],[] if unknown)."""
        f = self._by_name().get(frame_name)
        if f is None:
            return [], []
        return f["core_elements"], f["non_core_elements"]

    def is_non_core(self, frame_name: str, fe_name: str) -> bool:
        """SALSA/Sesame scoring: non-core (peripheral/extra-thematic) FEs = 0.5."""
        f = self._by_name().get(frame_name)
        return bool(f and fe_name in f["non_core_elements"])

    def frame_source(self, frame_name: str) -> str:
        """'FrameNet1.3' | 'SALSA' | 'SALSA-FrameNet1.2' | 'FrameNet1.2' | '?'."""
        f = self._by_name().get(frame_name)
        return f["source"] if f else "?"

    # -- trigger -> candidate frames --------------------------------------- #

    @lru_cache(1)
    def _candidate_maps(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """(lemma->frames, word->frames), normalised keys, order-preserving.

        Sources:
          * dictionary LUs  -> lemma key (e.g. 'abnehmer' -> [Abnehmer1-salsa])
          * training corpus -> both the gold target lemma and the trigger surface
            word map to the annotated frame.
        """
        lemma_map: dict[str, list[str]] = defaultdict(list)
        word_map: dict[str, list[str]] = defaultdict(list)

        def add(d: dict[str, list[str]], key: str, frame: str):
            if key and frame not in d[key]:
                d[key].append(frame)

        # (a) dictionary LUs
        for f in self.frames():
            for lu in f["lexical_units"]:
                add(lemma_map, _norm_word(lu), f["name"])

        # (b) empirical co-occurrences from the chosen (train) split(s).
        # Key the word map on the SINGLE whitespace word at the trigger loc —
        # exactly what candidate_frames() looks up at inference — so separable
        # verbs ('lehnen ab' target, 'lehnen' at loc) still match. Also add the
        # full joined trigger_text as a key for multiword lookups.
        for split in self.candidate_splits:
            for fi in iter_frame_instances(self.corpus_path, split=split):
                add(lemma_map, _norm_word(fi.trigger_lemma), fi.frame)
                add(word_map, _norm_word(_word_at(fi.text, fi.trigger_loc)), fi.frame)
                add(word_map, _norm_word(fi.trigger_text), fi.frame)

        return dict(lemma_map), dict(word_map)

    def candidate_frames_for_lemma(self, lemma: str) -> list[str]:
        lemma_map, _ = self._candidate_maps()
        return list(lemma_map.get(_norm_word(lemma), []))

    def candidate_frames_for_word(self, word: str) -> list[str]:
        _, word_map = self._candidate_maps()
        return list(word_map.get(_norm_word(word), []))

    def candidate_frames(
        self,
        text: str,
        trigger_loc: int,
        lemmatizer: Optional[Callable[[str], str]] = None,
    ) -> list[str]:
        """Candidate frames for the trigger word at `trigger_loc` in `text`.

        Dependency-free: matches the lowercased surface word (covers forms seen in
        training) and, if a `lemmatizer` callable is supplied, also its lemma —
        recommended for German inflection (e.g. simplemma/spaCy/HanTa). Returns a
        deduped, order-preserving candidate list (may be empty → mask is a no-op).
        """
        word = _word_at(text, trigger_loc)
        lemma_map, word_map = self._candidate_maps()
        out: list[str] = []
        for f in word_map.get(_norm_word(word), []):
            if f not in out:
                out.append(f)
        if lemmatizer is not None:
            try:
                lem = lemmatizer(word)
            except Exception:
                lem = None
            if lem:
                for f in lemma_map.get(_norm_word(lem), []):
                    if f not in out:
                        out.append(f)
        return out


# --------------------------------------------------------------------------- #
# whitespace-word helper (kept local so this module stays import-light)        #
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


def _word_at(text: str, loc: int) -> str:
    n = len(text)
    while loc < n and text[loc].isspace():
        loc += 1
    for s, e in _whitespace_words(text):
        if s <= loc < e:
            return text[s:e]
    return ""


# --------------------------------------------------------------------------- #
# Self-check                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import collections

    lx = SalsaLexicon()
    frames = lx.frames()
    print(f"frames defined      : {len(frames)}")
    print(f"frame_vocab (labels): {len(lx.frame_vocab())}")
    print(f"fe_vocab (roles)    : {len(lx.fe_vocab())}")

    src = collections.Counter(f["source"] for f in frames)
    print(f"sources             : {dict(src)}")
    empty_fe = sum(not (f['core_elements'] or f['non_core_elements']) for f in frames)
    print(f"frames w/ 0 FEs     : {empty_fe}")

    lemma_map, word_map = lx._candidate_maps()
    print(f"candidate lemma keys: {len(lemma_map)} | word keys: {len(word_map)}")

    # candidate-map coverage/ambiguity on the TEST split (recall ceiling for a
    # soft mask; how often the gold frame is among the candidates)
    from salsa_loader import load_frame_examples
    covered = total = 0
    cand_sizes = []
    for text, loc, gold in load_frame_examples("test"):
        cands = lx.candidate_frames(text, loc)
        total += 1
        cand_sizes.append(len(cands))
        if gold in cands:
            covered += 1
    print(f"\nTEST candidate coverage: {covered}/{total} = {100*covered/total:.1f}%"
          f"  (mean #candidates={sum(cand_sizes)/len(cand_sizes):.2f})")

    # a couple of concrete lookups
    print("\nexamples:")
    for w in ("applaudieren", "Kritiker", "sagen"):
        print(f"  {w!r:16} -> {lx.candidate_frames_for_word(w)[:6]}")
    ex = frames[0]
    print(f"\n  frame {ex['name']!r}: core={ex['core_elements']} "
          f"non_core={ex['non_core_elements'][:5]}...")
