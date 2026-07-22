"""
End-to-end German FrameParser: raw German text -> frame annotations.

The German counterpart of `src/texture_frames/pipeline.py`. Chains the three
stages on a sentence:

  1. trigger identification — the SALSA **lexicon rule** (salsa_trigger), NOT a
     neural head (see salsa_trigger for why partial annotation rules that out)
  2. frame classification — marker-pooled gbert model, candidate soft-masked
     (simplemma-lemmatized lexicon)
  3. argument extraction — detect-then-classify gbert model, FE-masked, NULL-bias

Operating points default to the dev-selected values from the trained runs
(frame candidate bias 4.0; args NULL bias 2.0; trigger rate threshold 0.5).

Load the two trained checkpoint dirs (as written by train_frame2_de /
train_args2_de — each holds `<head>_model.pt`, a `*_id.json`, and the tokenizer):

    from salsa_pipeline import GermanFrameParser
    parser = GermanFrameParser(
        frame_dir="/content/drive/MyDrive/Texture_Frames/models/frame2_de",
        args_dir ="/content/drive/MyDrive/Texture_Frames/models/args2_de",
    )
    for ann in parser.parse("Die Regierung kündigte an , die Steuern zu erhöhen ."):
        print(ann.frame, "|", ann.trigger, "|", [(a.role, a.text) for a in ann.arguments])
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field

# Shared, backbone-agnostic model classes live in the English tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from model_args2 import Args2Model  # noqa: E402
from model_frame2 import FrameMarkerModel  # noqa: E402

from args_data import _clean_span_text  # noqa: E402
from salsa_args_data import (  # noqa: E402
    NULL_ROLE,
    build_args_input,
    decode_detect_spans,
    frame_fe_hint,
)
from salsa_frame_data import (  # noqa: E402
    TRIGGER_END,
    TRIGGER_START,
    find_marker_positions,
    mark_trigger,
)
from salsa_lexicon import SalsaLexicon  # noqa: E402
from salsa_trigger import SalsaTriggerDetector  # noqa: E402


# --------------------------------------------------------------------------- #
# Output types (mirror the English pipeline)                                   #
# --------------------------------------------------------------------------- #

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


# --------------------------------------------------------------------------- #
# Helpers                                                                       #
# --------------------------------------------------------------------------- #

def _default_lemmatizer():
    try:
        import simplemma

        return lambda w: simplemma.lemmatize(w, lang="de")
    except Exception:
        return None


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
# Parser                                                                        #
# --------------------------------------------------------------------------- #

class GermanFrameParser:
    """Load once, then call `.parse(text)`. Loads the frame + args checkpoints
    from local dirs (e.g. mounted Google Drive); the trigger stage is a lexicon
    rule built from the corpus and needs no checkpoint."""

    def __init__(
        self,
        frame_dir: str,
        args_dir: str,
        device: str | None = None,
        frame_bias: float = 4.0,   # dev-picked candidate soft-mask strength
        null_bias: float = 2.0,    # dev-picked NULL-reject threshold for args
        trigger_threshold: float = 0.5,
        max_length: int = 320,
        lexicon: SalsaLexicon | None = None,
        trigger: SalsaTriggerDetector | None = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.max_length = max_length
        self.frame_bias = frame_bias
        self.null_bias = null_bias

        self.lexicon = lexicon or SalsaLexicon()
        self.lemmatizer = _default_lemmatizer()
        self.trigger_det = trigger or SalsaTriggerDetector(threshold=trigger_threshold)

        self.frame_model, self.frame_tok, self.frame2id, self.id2frame = self._load_frame(frame_dir)
        self.args_model, self.args_tok, self.role2id, self.id2role = self._load_args(args_dir)

        self._frame_start = self.frame_tok.convert_tokens_to_ids(TRIGGER_START)
        self._frame_end = self.frame_tok.convert_tokens_to_ids(TRIGGER_END)
        self._null_id = self.role2id[NULL_ROLE]

    # -- checkpoint loading ------------------------------------------------ #
    def _load_frame(self, d: str):
        meta = json.load(open(os.path.join(d, "frame2id.json")))
        frame2id = meta["frame2id"]
        tok = AutoTokenizer.from_pretrained(d)  # markers already added at train time
        model = FrameMarkerModel.from_pretrained(meta["base_model"], num_frames=len(frame2id))
        model.resize_token_embeddings(len(tok))
        model.load_state_dict(torch.load(os.path.join(d, "frame2_model.pt"), map_location=self.device))
        model.to(self.device).eval()
        return model, tok, frame2id, {int(i): f for f, i in frame2id.items()}

    def _load_args(self, d: str):
        meta = json.load(open(os.path.join(d, "role2id.json")))
        role2id = meta["role2id"]
        tok = AutoTokenizer.from_pretrained(d)
        model = Args2Model.from_pretrained(meta["base_model"], num_roles=len(role2id))
        model.resize_token_embeddings(len(tok))
        model.load_state_dict(torch.load(os.path.join(d, "args2_model.pt"), map_location=self.device))
        model.to(self.device).eval()
        return model, tok, role2id, {int(i): r for r, i in role2id.items()}

    # -- public ------------------------------------------------------------ #
    @torch.no_grad()
    def parse(self, text: str) -> list[FrameAnnotation]:
        """Parse a German sentence into frame annotations, one per detected
        trigger. Text is expected whitespace-tokenized (as SALSA/TIGER surfaces
        are), matching the trigger rule's word model."""
        annotations = []
        for loc in self.trigger_det.triggers(text):
            frame = self._frame(text, loc)
            args = self._args(text, loc, frame)
            annotations.append(
                FrameAnnotation(
                    trigger=_word_at(text, loc), trigger_loc=loc, frame=frame, arguments=args
                )
            )
        return annotations

    # -- stages ------------------------------------------------------------ #
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
            span_text = _clean_span_text(combined[cs:ce])
            start = text.find(span_text)
            args.append(Argument(
                role=self.id2role[r], text=span_text,
                start=start, end=start + len(span_text) if start >= 0 else -1,
            ))
        return args


# --------------------------------------------------------------------------- #
# CLI                                                                           #
# --------------------------------------------------------------------------- #

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Parse German text into frame annotations.")
    p.add_argument("--frame-dir", required=True, help="dir of the trained frame checkpoint")
    p.add_argument("--args-dir", required=True, help="dir of the trained args checkpoint")
    p.add_argument("--device", default=None)
    p.add_argument("text", nargs="+", help="sentence(s) to parse")
    a = p.parse_args()

    parser = GermanFrameParser(frame_dir=a.frame_dir, args_dir=a.args_dir, device=a.device)
    for sent in a.text:
        print(f"\n# {sent}")
        for ann in parser.parse(sent):
            print(f"[{ann.frame}] trigger={ann.trigger!r}")
            for arg in ann.arguments:
                print(f"    {arg.role:16} {arg.text!r}")
