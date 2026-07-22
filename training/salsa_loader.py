"""
SALSA 2.0 (SALSA/TIGER XML) data loader — the German counterpart of
`texture_frames.data` / `texture_frames.args_data`.

It reads the native SALSA release and yields the *same* tuple shapes the English
encoder heads consume, so the German heads can reuse the existing training code
with only a backbone/lexicon swap:

    load_trigger_sentences(split) -> [(text, [trigger_loc, ...]), ...]
    load_frame_examples(split)    -> [(text, trigger_loc, frame), ...]
    load_args_examples(split)     -> [(text, trigger_loc, frame,
                                       [(role, start_char, end_char), ...]), ...]

`text` is a surface string reconstructed by space-joining a sentence's TIGER
terminals; every char offset (trigger_loc, role spans) indexes into that string,
matching the whitespace-word model the English parser already uses.

Design notes specific to SALSA (see the feasibility findings):
  * Role/target spans are given as *syntax-tree nodes* (`<fenode idref>`), either
    a terminal (single token) or a nonterminal (a constituent). Nonterminal yields
    are resolved recursively via `<edge idref>` and mapped to an enclosing char
    span (min-start .. max-end). Discontinuous yields are flagged.
  * SALSA annotates only ~685 target *lemmas* across TIGER, so a sentence's
    trigger set is NOT exhaustive. `load_trigger_sentences` is provided for
    completeness but carries this caveat — see README / feasibility notes.
  * ~42% of frame instances use SALSA lemma-specific "proto-frames"; `source` for
    each frame name is available via `frame_sources()` if you want to filter.
  * Underspecification (`<usp>`, co-applying frames on one target) is kept as
    independent frame instances by default; pass `drop_underspecified=True` to
    skip targets that participate in a `<uspframes>` block.

Pure stdlib (xml.etree + hashlib) — no torch/nltk/transformers needed to load.
"""
from __future__ import annotations

import hashlib
import os
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Iterator, Optional

# --------------------------------------------------------------------------- #
# Locations                                                                    #
# --------------------------------------------------------------------------- #

_HERE = os.path.dirname(os.path.abspath(__file__))
# The single-file corpus: each TIGER sentence once, all lemma annotations merged.
DEFAULT_CORPUS = os.path.join(_HERE, "extracted", "salsa_release.xml")
DEFAULT_FRAMES = os.path.join(_HERE, "extracted", "salsa_frames.xml")

# Deterministic sentence-level split proportions (train, dev, test).
_SPLIT_BUCKETS = 100
_DEV_HI = 80          # buckets [0,80)  -> train
_TEST_HI = 90         # buckets [80,90) -> dev, [90,100) -> test

# SALSA pseudo-frame: a target the annotators judged to evoke NO appropriate
# frame. Real label in the data (~117 instances), but usually excluded when
# training a frame-evoking parser — pass drop_unannotated=True.
UNANNOTATED_FRAME = "Unannotated"


# --------------------------------------------------------------------------- #
# Structured records                                                           #
# --------------------------------------------------------------------------- #

@dataclass
class RoleSpan:
    role: str
    start: int            # char offset into the reconstructed sentence text
    end: int
    discontinuous: bool = False


@dataclass
class FrameInstance:
    sentence_id: str
    text: str
    trigger_loc: int      # char offset of the (first) target token
    trigger_text: str
    trigger_lemma: str    # the target's lemma (from <target lemma=...>); "" if absent
    frame: str
    roles: list = field(default_factory=list)   # list[RoleSpan]
    target_is_nonterminal: bool = False
    underspecified: bool = False


# --------------------------------------------------------------------------- #
# Namespace-robust helpers                                                     #
# --------------------------------------------------------------------------- #

def _local(tag: str) -> str:
    """Strip any `{namespace}` prefix from an ElementTree tag."""
    return tag.rsplit("}", 1)[-1]


def _split_of(sentence_id: str) -> str:
    """Stable train/dev/test assignment from a sentence id (reproducible across
    runs — Python's str hash is randomized, so we use md5)."""
    h = int(hashlib.md5(sentence_id.encode("utf-8")).hexdigest(), 16) % _SPLIT_BUCKETS
    if h < _DEV_HI:
        return "train"
    if h < _TEST_HI:
        return "dev"
    return "test"


# --------------------------------------------------------------------------- #
# Per-sentence parsing                                                         #
# --------------------------------------------------------------------------- #

class _Sentence:
    """Reconstructs surface text + char offsets and resolves fenode -> char span
    for one `<s>` element."""

    def __init__(self, s_elem: ET.Element):
        self.id = s_elem.get("id", "")
        # ordered terminals
        self._term_ids: list[str] = []
        self._word: dict[str, str] = {}
        self._lemma: dict[str, str] = {}
        self._pos: dict[str, str] = {}
        # nonterminal -> list of child idrefs (terminals or nonterminals)
        self._nt_edges: dict[str, list[str]] = {}

        graph = None
        for child in s_elem:
            if _local(child.tag) == "graph":
                graph = child
                break
        if graph is not None:
            for sub in graph:
                lname = _local(sub.tag)
                if lname == "terminals":
                    for t in sub:
                        if _local(t.tag) != "t":
                            continue
                        tid = t.get("id")
                        if tid is None:
                            continue
                        self._term_ids.append(tid)
                        self._word[tid] = t.get("word", "")
                        self._lemma[tid] = t.get("lemma", "")
                        self._pos[tid] = t.get("pos", "")
                elif lname == "nonterminals":
                    for nt in sub:
                        if _local(nt.tag) != "nt":
                            continue
                        ntid = nt.get("id")
                        if ntid is None:
                            continue
                        self._nt_edges[ntid] = [
                            e.get("idref")
                            for e in nt
                            if _local(e.tag) == "edge" and e.get("idref")
                        ]

        # surface text (space-joined terminals) + per-terminal char span
        self.text_parts: list[str] = []
        self.span_of: dict[str, tuple[int, int]] = {}
        self.index_of: dict[str, int] = {i_id: i for i, i_id in enumerate(self._term_ids)}
        pos = 0
        for i, tid in enumerate(self._term_ids):
            if i > 0:
                pos += 1  # the joining space
            w = self._word[tid]
            self.span_of[tid] = (pos, pos + len(w))
            self.text_parts.append(w)
            pos += len(w)
        self.text = " ".join(self.text_parts)

    # -- yield resolution -------------------------------------------------- #

    def terminal_yield(self, node_id: str, _seen: Optional[set] = None) -> list[str]:
        """All terminal ids dominated by node_id (a terminal or nonterminal),
        in surface order. Cycle-safe."""
        if node_id in self.span_of:  # it's a terminal
            return [node_id]
        if _seen is None:
            _seen = set()
        if node_id in _seen or node_id not in self._nt_edges:
            return []
        _seen.add(node_id)
        out: list[str] = []
        for child in self._nt_edges[node_id]:
            out.extend(self.terminal_yield(child, _seen))
        # de-dup, keep surface order
        uniq = {t: None for t in out}
        return sorted(uniq, key=lambda t: self.index_of.get(t, 1 << 30))

    def node_char_span(self, node_id: str) -> Optional[tuple[int, int, bool]]:
        """(start_char, end_char, discontinuous) enclosing a node's terminal
        yield, or None if it resolves to nothing."""
        terms = self.terminal_yield(node_id)
        if not terms:
            return None
        idxs = sorted(self.index_of[t] for t in terms if t in self.index_of)
        starts = [self.span_of[t][0] for t in terms if t in self.span_of]
        ends = [self.span_of[t][1] for t in terms if t in self.span_of]
        if not starts:
            return None
        discontinuous = idxs != list(range(idxs[0], idxs[-1] + 1))
        return min(starts), max(ends), discontinuous

    def fenodes_char_span(self, fenodes: list[str]) -> Optional[tuple[int, int, bool]]:
        """Enclosing char span over a (possibly multi-fenode) FE/target."""
        spans = [self.node_char_span(fn) for fn in fenodes]
        spans = [sp for sp in spans if sp]
        if not spans:
            return None
        start = min(sp[0] for sp in spans)
        end = max(sp[1] for sp in spans)
        disc = len(spans) > 1 or any(sp[2] for sp in spans)
        return start, end, disc

    def is_terminal(self, node_id: str) -> bool:
        return node_id in self.span_of

    def target_head_start(self, fenodes: list[str]) -> Optional[int]:
        """Char start of the target's *content head* token, the anchor the heads
        train on. For single-token targets (97%) this is that token. For
        multiword targets (separable verbs, particle+verb, nominal MWEs) it picks
        the content head — a full/finite verb, else another verb, else a
        noun/adjective — rather than the leftmost token, so we never anchor the
        trigger on a preposition or separable particle (e.g. 'zu brennen' -> the
        verb, 'schlägt … vor' -> the finite verb, not the particle).
        Uses TIGER STTS part-of-speech tags."""
        terms: list[str] = []
        for fn in fenodes:
            terms.extend(self.terminal_yield(fn))
        terms = sorted({t for t in terms if t in self.index_of}, key=self.index_of.get)
        if not terms:
            return None

        def rank(t: str) -> int:
            p = self._pos.get(t, "")
            if p.startswith(("VV", "VM", "VA")):   # full / modal / auxiliary verb
                return 0
            if p.startswith("V"):                  # any other verb tag
                return 1
            if p.startswith(("NN", "NE", "ADJ")):  # noun / proper noun / adjective
                return 2
            if p in ("PTKVZ", "PTKZU", "APPR", "APPRART", "ART", "KOUS"):
                return 5                            # particles/prepositions: avoid
            return 3

        best = min(terms, key=lambda t: (rank(t), self.index_of[t]))
        return self.span_of[best][0]

    def node_words(self, fenodes: list[str]) -> str:
        """Surface words of the terminals a set of fenodes dominates, in order —
        honest text for discontinuous targets (e.g. separable verb 'lehnen ab'),
        unlike the enclosing char span which would swallow intervening tokens."""
        terms: list[str] = []
        for fn in fenodes:
            terms.extend(self.terminal_yield(fn))
        uniq = sorted({t for t in terms if t in self.index_of}, key=self.index_of.get)
        return " ".join(self._word[t] for t in uniq)


# --------------------------------------------------------------------------- #
# Corpus iteration                                                             #
# --------------------------------------------------------------------------- #

def iter_frame_instances(
    corpus_path: str = DEFAULT_CORPUS,
    split: Optional[str] = None,
    drop_underspecified: bool = False,
    drop_unannotated: bool = False,
) -> Iterator[FrameInstance]:
    """Stream every frame instance in the corpus as a `FrameInstance`.

    split ∈ {None, "train", "dev", "test"}; None yields all.
    drop_unannotated skips the `Unannotated` pseudo-frame (see UNANNOTATED_FRAME).
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(
            f"SALSA corpus not found at {corpus_path!r}. Point corpus_path at the "
            f"extracted salsa_release.xml."
        )

    for _event, s_elem in ET.iterparse(corpus_path, events=("end",)):
        if _local(s_elem.tag) != "s":
            continue
        sid = s_elem.get("id", "")
        if split is not None and _split_of(sid) != split:
            s_elem.clear()
            continue

        sent = _Sentence(s_elem)

        # locate <sem>/<frames> and <usp>/<uspframes>
        sem = None
        for c in s_elem:
            if _local(c.tag) == "sem":
                sem = c
                break
        if sem is None:
            s_elem.clear()
            continue

        usp_frame_ids: set[str] = set()
        frames_elem = None
        for c in sem:
            lname = _local(c.tag)
            if lname == "frames":
                frames_elem = c
            elif lname == "usp":
                for u in c:
                    if _local(u.tag) == "uspframes":
                        for blk in u:
                            for fid in blk.iter():
                                ref = fid.get("idref")
                                if ref:
                                    usp_frame_ids.add(ref)

        if frames_elem is None:
            s_elem.clear()
            continue

        for fr in frames_elem:
            if _local(fr.tag) != "frame":
                continue
            fname = fr.get("name")
            fid = fr.get("id", "")
            if not fname:
                continue
            if drop_unannotated and fname == UNANNOTATED_FRAME:
                continue
            is_usp = fid in usp_frame_ids
            if drop_underspecified and is_usp:
                continue

            target = None
            for c in fr:
                if _local(c.tag) == "target":
                    target = c
                    break
            if target is None:
                continue
            tgt_fenodes = [
                fn.get("idref") for fn in target
                if _local(fn.tag) == "fenode" and fn.get("idref")
            ]
            if not tgt_fenodes:
                continue
            tgt_span = sent.fenodes_char_span(tgt_fenodes)
            if tgt_span is None:
                continue
            # anchor on the content head token, not the leftmost — see
            # _Sentence.target_head_start. Falls back to span start if unresolved.
            trigger_loc = sent.target_head_start(tgt_fenodes)
            if trigger_loc is None:
                trigger_loc = tgt_span[0]
            target_is_nt = not all(sent.is_terminal(fn) for fn in tgt_fenodes)
            trigger_text = sent.node_words(tgt_fenodes)
            trigger_lemma = target.get("lemma", "") or ""

            roles: list[RoleSpan] = []
            for fe in fr:
                if _local(fe.tag) != "fe":
                    continue
                rname = fe.get("name")
                if not rname:
                    continue
                fe_fenodes = [
                    fn.get("idref") for fn in fe
                    if _local(fn.tag) == "fenode" and fn.get("idref")
                ]
                if not fe_fenodes:
                    continue  # unrealized (null-instantiated) FE — no span
                sp = sent.fenodes_char_span(fe_fenodes)
                if sp is None:
                    continue
                roles.append(RoleSpan(role=rname, start=sp[0], end=sp[1],
                                      discontinuous=sp[2]))

            yield FrameInstance(
                sentence_id=sid,
                text=sent.text,
                trigger_loc=trigger_loc,
                trigger_text=trigger_text,
                trigger_lemma=trigger_lemma,
                frame=fname,
                roles=roles,
                target_is_nonterminal=target_is_nt,
                underspecified=is_usp,
            )

        s_elem.clear()


# --------------------------------------------------------------------------- #
# Public loaders — mirror texture_frames.data / args_data                      #
# --------------------------------------------------------------------------- #

def load_frame_examples(
    split: str, corpus_path: str = DEFAULT_CORPUS, **kw
) -> list[tuple[str, int, str]]:
    """[(text, trigger_loc, frame), ...] — mirrors data.load_frame_examples."""
    return [
        (fi.text, fi.trigger_loc, fi.frame)
        for fi in iter_frame_instances(corpus_path, split=split, **kw)
    ]


def load_args_examples(
    split: str, corpus_path: str = DEFAULT_CORPUS, keep_discontinuous: bool = True, **kw
) -> list[tuple[str, int, str, list[tuple[str, int, int]]]]:
    """[(text, trigger_loc, frame, [(role, start, end), ...]), ...] —
    mirrors args_data.load_args_examples. Set keep_discontinuous=False to drop
    role spans whose constituent yield is non-contiguous in surface order."""
    out = []
    for fi in iter_frame_instances(corpus_path, split=split, **kw):
        fes = [
            (r.role, r.start, r.end)
            for r in fi.roles
            if keep_discontinuous or not r.discontinuous
        ]
        out.append((fi.text, fi.trigger_loc, fi.frame, fes))
    return out


def load_trigger_sentences(
    split: str, corpus_path: str = DEFAULT_CORPUS, **kw
) -> list[tuple[str, list[int]]]:
    """[(text, [trigger_loc, ...]), ...] — mirrors data.load_trigger_sentences.

    CAVEAT: SALSA is not exhaustively annotated (only ~685 target lemmas), so the
    returned trigger set per sentence is a LOWER BOUND on frame-evoking words.
    Training a closed-world trigger tagger on this mislabels unannotated
    predicates as negatives — see feasibility notes before using this head.
    """
    by_sent: dict[str, tuple[str, set[int]]] = {}
    for fi in iter_frame_instances(corpus_path, split=split, **kw):
        text, locs = by_sent.setdefault(fi.sentence_id, (fi.text, set()))
        locs.add(fi.trigger_loc)
    return [(text, sorted(locs)) for text, locs in by_sent.values()]


# --------------------------------------------------------------------------- #
# Lexicon-adjacent helper                                                      #
# --------------------------------------------------------------------------- #

def frame_sources(frames_path: str = DEFAULT_FRAMES) -> dict[str, str]:
    """frame name -> source ('FrameNet1.3' | 'SALSA' | 'SALSA-FrameNet1.2' |
    'FrameNet1.2'). Lets callers filter/keep proto-frames.

    salsa_frames.xml is NOT strict XML (it carries unescaped `&` in example
    prose, e.g. "Laut 1&1 …"), so this reads it leniently with a regex over the
    `<frame …>` open tags rather than a validating parser."""
    import re

    text = open(frames_path, encoding="utf-8", errors="replace").read()
    out: dict[str, str] = {}
    for m in re.finditer(r"<frame\b([^>]*)>", text):
        attrs = m.group(1)
        name = re.search(r"\bname=['\"]([^'\"]+)", attrs)
        if not name:
            continue
        src = re.search(r"\bsource=['\"]([^'\"]+)", attrs)
        out[name.group(1)] = src.group(1) if src else "?"
    return out


# --------------------------------------------------------------------------- #
# Self-check                                                                    #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import collections
    import sys

    corpus = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CORPUS
    print(f"corpus: {corpus}\n")

    n = n_roles = n_disc = n_nt_target = n_usp = 0
    frames = collections.Counter()
    split_counts = collections.Counter()
    samples: list[FrameInstance] = []
    for fi in iter_frame_instances(corpus):
        n += 1
        n_roles += len(fi.roles)
        n_disc += sum(r.discontinuous for r in fi.roles)
        n_nt_target += fi.target_is_nonterminal
        n_usp += fi.underspecified
        frames[fi.frame] += 1
        split_counts[_split_of(fi.sentence_id)] += 1
        if len(samples) < 3 and fi.roles:
            samples.append(fi)

    print(f"frame instances : {n}")
    print(f"role spans      : {n_roles}  (discontinuous: {n_disc}, "
          f"{100*n_disc/max(n_roles,1):.1f}%)")
    print(f"nonterminal targets: {n_nt_target}  ({100*n_nt_target/max(n,1):.1f}%)")
    print(f"underspecified frames: {n_usp}")
    print(f"distinct frames : {len(frames)}")
    print(f"split (frame instances): {dict(split_counts)}")
    print("\n=== sample decodes ===")
    for fi in samples:
        print(f"\n[{fi.frame}]  trigger={fi.trigger_text!r} @ {fi.trigger_loc}")
        print(f"  text: {fi.text[:140]}")
        for r in fi.roles:
            tag = " (disc)" if r.discontinuous else ""
            print(f"    {r.role:20} {fi.text[r.start:r.end]!r}{tag}")
