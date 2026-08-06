"""
End-to-end German FrameParser: raw text -> frame annotations.

Chains the three stages on a sentence:
  1. trigger identification — the SALSA lexicon rule (trigger.TriggerDetector)
  2. frame classification — marker-pooled gbert model, candidate soft-masked
  3. argument extraction — detect-then-classify gbert model, FE-masked, NULL-bias

Operating points default to the dev-selected values from training (frame
candidate bias 4.0; args NULL bias 2.0; trigger rate threshold 0.5).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import torch

from . import weights
from ._args_helpers import (
    NULL_ROLE,
    build_args_input,
    clean_span_text,
    decode_detect_spans,
    frame_fe_hint,
)
from ._tokenization import (
    TRIGGER_END,
    TRIGGER_START,
    find_marker_positions,
    mark_trigger,
    word_at,
)
from .lexicon import Lexicon, default_lemmatizer
from .trigger import TriggerDetector, content_verb_locs

DEFAULT_FRAME_REPO = "texturejc/texture-frames-de-frame"
DEFAULT_ARGS_REPO = "texturejc/texture-frames-de-args"

_DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


@dataclass
class Argument:
    role: str
    text: str
    start: int  # char offset in the sentence (-1 if not locatable)
    end: int


@dataclass
class FrameAnnotation:
    trigger: str
    trigger_loc: int
    frame: str
    arguments: list = field(default_factory=list)


class FrameParser:
    """Load once, then call `.parse(text)`. The frame + args weights download
    from the HF Hub on first construction (cached by huggingface_hub); the
    trigger stage and lexicon are bundled with the package."""

    def __init__(
        self,
        device: str | None = None,
        frame_repo: str = DEFAULT_FRAME_REPO,
        args_repo: str = DEFAULT_ARGS_REPO,
        frame_bias: float = 4.0,   # dev-picked candidate soft-mask strength
        null_bias: float = 2.0,    # dev-picked NULL-reject threshold for args
        trigger_threshold: float = 0.5,
        max_length: int = 320,
        open_vocab: bool = False,  # detect out-of-lexicon verbs via POS + prior-correct
        prior_alpha: float = 0.75,  # class-prior correction strength for OOV triggers
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.frame_bias = frame_bias
        self.null_bias = null_bias
        self.open_vocab = open_vocab
        self.prior_alpha = prior_alpha

        self.lexicon = Lexicon()
        self.lemmatizer = default_lemmatizer()
        self.trigger_det = TriggerDetector(threshold=trigger_threshold)

        self.frame_model, self.frame_tok, self.frame2id, self.id2frame = weights.load_frame(
            frame_repo, self.device
        )
        self.args_model, self.args_tok, self.role2id, self.id2role = weights.load_args(
            args_repo, self.device
        )
        self._frame_start = self.frame_tok.convert_tokens_to_ids(TRIGGER_START)
        self._frame_end = self.frame_tok.convert_tokens_to_ids(TRIGGER_END)
        self._null_id = self.role2id[NULL_ROLE]

        # log class-prior over the frame label space (from train counts) — used to
        # de-bias the frame head for OOV triggers, which have no candidate mask.
        counts = json.load(open(os.path.join(_DATA, "frame_counts.json"), encoding="utf-8"))
        tot, n = sum(counts.values()), len(self.frame2id)
        self._logprior = torch.tensor(
            [math.log((counts.get(f, 0) + 1) / (tot + n))
             for f, _ in sorted(self.frame2id.items(), key=lambda kv: kv[1])],
            device=self.device,
        )

    # -- public ------------------------------------------------------------ #
    @torch.no_grad()
    def parse(self, text: str) -> list[FrameAnnotation]:
        """Parse a single German sentence into frame annotations, one per
        detected trigger. Text should be whitespace-tokenized (punctuation
        spaced out), as SALSA/TIGER surfaces are. For multi-sentence input,
        use `parse_document`."""
        annotations = []
        for loc in self._triggers(text):
            frame = self._frame(text, loc)
            args = self._args(text, loc, frame)
            annotations.append(
                FrameAnnotation(
                    trigger=word_at(text, loc), trigger_loc=loc, frame=frame, arguments=args
                )
            )
        return annotations

    def parse_document(self, text: str) -> list[FrameAnnotation]:
        """Sentence-split `text` (NLTK Punkt, German) and run `parse` per
        sentence. Returns a flat list of `FrameAnnotation` with `trigger_loc`
        and `Argument.start`/`end` shifted to offsets in the original document.

        Requires the optional `nltk` dependency:
            pip install "texture-frames-de[sentencize]"
        """
        try:
            import nltk
            from nltk.tokenize import PunktTokenizer
        except ImportError as e:
            raise ImportError(
                "parse_document requires nltk. Install with: "
                'pip install "texture-frames-de[sentencize]"'
            ) from e
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.download("punkt_tab", quiet=True)

        tokenizer = PunktTokenizer("german")
        annotations = []
        for sent_start, sent_end in tokenizer.span_tokenize(text):
            for ann in self.parse(text[sent_start:sent_end]):
                ann.trigger_loc += sent_start
                for arg in ann.arguments:
                    if arg.start >= 0:
                        arg.start += sent_start
                        arg.end += sent_start
                annotations.append(ann)
        return annotations

    # -- stages ------------------------------------------------------------ #
    def _triggers(self, text: str) -> list[int]:
        """Trigger char offsets: the lexicon rule, plus (open_vocab) POS-detected
        content verbs the lexicon missed. Deduped by whitespace word, in order."""
        locs = set(self.trigger_det.triggers(text))
        if self.open_vocab:
            locs |= set(content_verb_locs(text))
        return sorted(locs)

    def _frame(self, text: str, loc: int) -> str:
        enc = self.frame_tok(
            mark_trigger(text, loc), truncation=True, max_length=self.max_length,
            return_tensors="pt",
        )
        sp, ep = find_marker_positions(enc["input_ids"][0].tolist(), self._frame_start, self._frame_end)
        logits = self.frame_model.encode_logits(
            enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device),
            torch.tensor([sp], device=self.device), torch.tensor([ep], device=self.device),
        )[0]
        cand_ids = [self.frame2id[c] for c in
                    self.lexicon.candidate_frames(text, loc, lemmatizer=self.lemmatizer)
                    if c in self.frame2id]
        if cand_ids:
            logits = logits.clone()
            logits[torch.tensor(cand_ids, device=self.device)] += self.frame_bias
        elif self.open_vocab and self.prior_alpha:
            # OOV trigger: no candidate mask available. De-bias the frame head with
            # the class prior so its genuine (rank-1/2) semantic signal wins over
            # high-frequency generic frames.
            logits = logits - self.prior_alpha * self._logprior
        return self.id2frame[int(logits.argmax())]

    def _allowed_role_ids(self, frame: str):
        core, non_core = self.lexicon.frame_elements(frame)
        allowed = {self._null_id}
        for fe in [*core, *non_core]:
            if fe in self.role2id:
                allowed.add(self.role2id[fe])
        return allowed

    def _args(self, text: str, loc: int, frame: str) -> list[Argument]:
        hint = frame_fe_hint(self.lexicon, frame)
        combined, prefix_len, _, _ = build_args_input(text, frame, loc, hint)
        enc = self.args_tok(
            combined, truncation=True, max_length=self.max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        hidden, detect_logits = self.args_model.encode(
            enc["input_ids"].to(self.device), enc["attention_mask"].to(self.device)
        )
        spans = decode_detect_spans(offsets, detect_logits[0].argmax(-1).tolist(), prefix_len)
        if not spans:
            return []

        mask = torch.full((len(self.role2id),), float("-inf"), device=self.device)
        mask[list(self._allowed_role_ids(frame))] = 0.0
        role_logits = self.args_model.role_logits_for_spans(
            hidden, [(0, s, e) for (s, e, _, _) in spans]
        ) + mask
        role_logits[:, self._null_id] += self.null_bias

        args = []
        for (_, _, cs, ce), r in zip(spans, role_logits.argmax(-1).tolist()):
            if r == self._null_id:
                continue
            span_text = clean_span_text(combined[cs:ce])
            start = text.find(span_text)
            args.append(Argument(
                role=self.id2role[r], text=span_text,
                start=start, end=start + len(span_text) if start >= 0 else -1,
            ))
        return args
