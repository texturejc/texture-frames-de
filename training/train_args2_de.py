"""
Train the German argument-extraction v2 detect-then-classify model on SALSA.

The German counterpart of `encoder_parser/train_args2.py`. Same model
(`Args2Model`, reused unchanged — backbone-agnostic) and same detect-then-classify
recipe; only the data source, lexicon, and backbone differ:

  * backbone : deepset/gbert-large      (was microsoft/deberta-v3-large)
  * data     : salsa_args_data          (was args2_data / NLTK FrameNet)
  * lexicon  : SalsaLexicon             (was Lexicon)

No synonym augmentation (the English WordNet path has no drop-in German
equivalent here). `--keep-discontinuous/--drop-discontinuous` toggles whether the
13.8% discontinuous-yield role spans are trained on (kept by default).

Run:
    python train_args2_de.py --epochs 5 --batch-size 16
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import json
import sys
import time

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

# Reuse the shared, backbone-agnostic model + pure eval helpers from the English tree.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_args2 import IGNORE_INDEX, Args2Model  # noqa: E402
from args_data import _clean_span_text, score_args  # noqa: E402

from salsa_args_data import (  # noqa: E402
    NULL_ROLE,
    build_args2_dataset,
    build_args_input,
    decode_detect_spans,
    frame_fe_hint,
    role_label_maps,
)
from salsa_lexicon import SalsaLexicon  # noqa: E402
from salsa_loader import load_args_examples  # noqa: E402

# Markers already live in the tokenizer via the frame head's convention.
TRIGGER_START = "<t>"
TRIGGER_END = "</t>"

DEFAULT_BASE_MODEL = "deepset/gbert-large"
DEFAULT_NULL_BIASES = [-4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0]


# --------------------------------------------------------------------------- #
# Collator + Trainer (identical to the English args2 trainer)                  #
# --------------------------------------------------------------------------- #

class Args2Collator:
    """Right-pad input_ids/attention_mask/detect_labels; carry `spans` as a plain
    Python list (token indices stay valid because padding is on the right)."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, detect, spans = [], [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            detect.append(f["detect_labels"] + [IGNORE_INDEX] * pad)
            spans.append([tuple(s) for s in f["spans"]])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "detect_labels": torch.tensor(detect, dtype=torch.long),
            "spans": spans,
        }


class Args2Trainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)
        loss = outputs["loss"]
        return (loss, outputs) if return_outputs else loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # eval: only the loss — `spans` is a ragged Python list, not gatherable
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs)
        return (loss.detach(), None, None)


# --------------------------------------------------------------------------- #
# German test eval (mirrors eval_args2, no English-baseline references)        #
# --------------------------------------------------------------------------- #

def _allowed_role_ids(frame, lexicon, role2id):
    core, non_core = lexicon.frame_elements(frame)
    allowed = {role2id[NULL_ROLE]}
    for fe in [*core, *non_core]:
        if fe in role2id:
            allowed.add(role2id[fe])
    return allowed


@torch.no_grad()
def evaluate_args2_de(
    model, tokenizer, lexicon, role2id, id2role,
    split="test", max_length=320, null_biases=None, keep_discontinuous=True,
):
    null_biases = list(null_biases) if null_biases is not None else DEFAULT_NULL_BIASES
    model.eval()
    device = next(model.parameters()).device
    num_roles = len(role2id)
    null_id = role2id[NULL_ROLE]
    mask_cache: dict[str, torch.Tensor] = {}

    def role_mask(frame):
        if frame not in mask_cache:
            m = torch.full((num_roles,), float("-inf"), device=device)
            m[list(_allowed_role_ids(frame, lexicon, role2id))] = 0.0
            mask_cache[frame] = m
        return mask_cache[frame]

    examples = load_args_examples(
        split, keep_discontinuous=keep_discontinuous, drop_unannotated=True
    )
    tp = {b: 0.0 for b in null_biases}
    fp = {b: 0.0 for b in null_biases}
    fn = {b: 0.0 for b in null_biases}
    t0 = time.time()
    for text, trigger_loc, frame, gold_fes in examples:
        gold_spans = [(name, _clean_span_text(text[s:e])) for name, s, e in gold_fes]
        is_nc = lambda fe: lexicon.is_non_core(frame, fe)  # noqa: E731

        hint = frame_fe_hint(lexicon, frame)
        combined, prefix_len, _, _ = build_args_input(text, frame, trigger_loc, hint)
        enc = tokenizer(
            combined, truncation=True, max_length=max_length,
            return_offsets_mapping=True, return_tensors="pt",
        )
        offsets = enc["offset_mapping"][0].tolist()
        hidden, detect_logits = model.encode(
            enc["input_ids"].to(device), enc["attention_mask"].to(device)
        )
        spans = decode_detect_spans(offsets, detect_logits[0].argmax(-1).tolist(), prefix_len)

        role_logits = None
        if spans:
            span_bi = [(0, s_tok, e_tok) for (s_tok, e_tok, _, _) in spans]
            role_logits = model.role_logits_for_spans(hidden, span_bi) + role_mask(frame)

        for b in null_biases:
            pred_spans = []
            if role_logits is not None:
                biased = role_logits.clone()
                biased[:, null_id] += b
                for (_, _, cs, ce), r in zip(spans, biased.argmax(-1).tolist()):
                    if r != null_id:
                        pred_spans.append((id2role[r], _clean_span_text(combined[cs:ce])))
            s_tp, s_fp, s_fn = score_args(gold_spans, pred_spans, is_nc)
            tp[b] += s_tp
            fp[b] += s_fp
            fn[b] += s_fn
    elapsed = time.time() - t0

    by_bias = {}
    for b in null_biases:
        p = tp[b] / (tp[b] + fp[b]) if (tp[b] + fp[b]) else 0.0
        r = tp[b] / (tp[b] + fn[b]) if (tp[b] + fn[b]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        by_bias[b] = {"precision": p, "recall": r, "f1": f1,
                      "tp": tp[b], "fp": fp[b], "fn": fn[b]}
    return {
        "by_bias": by_bias,
        "n_examples": len(examples),
        "ms_per_example": 1000 * elapsed / max(len(examples), 1),
        "split": split,
    }


def print_report(metrics):
    by_bias = metrics["by_bias"]
    best_b = max(by_bias, key=lambda b: by_bias[b]["f1"])
    print("=" * 60)
    print(f"Argument extraction v2 (German / SALSA) — {metrics['split']} split")
    print("=" * 60)
    print(f"{'null_bias':<11}{'P':>8}{'R':>8}{'F1':>8}")
    print("-" * 40)
    for b, m in by_bias.items():
        star = "  <- best" if b == best_b else ""
        print(f"{b:<11.1f}{m['precision']:>8.3f}{m['recall']:>8.3f}{m['f1']:>8.3f}{star}")
    print("-" * 40)
    bm = by_bias[best_b]
    print(f"best null_bias {best_b:+.1f}: F1 {bm['f1']:.3f} "
          f"(tp/fp/fn {bm['tp']:.1f}/{bm['fp']:.1f}/{bm['fn']:.1f}, non-core 0.5)")
    print(f"speed {metrics['ms_per_example']:.2f} ms/example over {metrics['n_examples']} examples")
    print("NOTE: pick the bias on DEV, then report it on TEST.")
    print("=" * 60)


# --------------------------------------------------------------------------- #
# Train                                                                        #
# --------------------------------------------------------------------------- #

def train(
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = "outputs/args2_de",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 320,
    role_lambda: float = 1.0,
    n_negatives: int = 4,
    keep_discontinuous: bool = True,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    resume: bool = True,
):
    lexicon = SalsaLexicon()
    fe_vocab = lexicon.fe_vocab()
    roles, role2id, id2role = role_label_maps(fe_vocab)
    print(f"roles: {len(roles)} (NULL + {len(fe_vocab)} FEs) | German / SALSA")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    assert tokenizer.is_fast, "need a fast tokenizer for offset_mapping"
    tokenizer.add_special_tokens({"additional_special_tokens": [TRIGGER_START, TRIGGER_END]})

    model = Args2Model.from_pretrained(base_model, num_roles=len(roles), role_lambda=role_lambda)
    model.resize_token_embeddings(len(tokenizer))

    train_ds = build_args2_dataset(
        "train", tokenizer, role2id, lexicon, max_length, n_negatives, keep_discontinuous
    )
    dev_ds = build_args2_dataset(
        "dev", tokenizer, role2id, lexicon, max_length, n_negatives, keep_discontinuous
    )
    print(f"train examples: {len(train_ds)}   dev examples: {len(dev_ds)}")

    collator = Args2Collator(tokenizer)
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=lr,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        bf16=use_bf16,
        fp16=use_fp16,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        save_total_limit=2,  # keep best + most-recent so a disconnect can resume
        remove_unused_columns=False,  # keep detect_labels/spans for the model
        label_names=["detect_labels", "spans"],
        report_to="none",
    )

    trainer = Args2Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=collator,
    )

    # Auto-resume from the last checkpoint in output_dir (Drive) if present, so a
    # Colab disconnect can be continued by simply re-running the train cell.
    last_ckpt = None
    if resume and os.path.isdir(output_dir):
        from transformers.trainer_utils import get_last_checkpoint

        last_ckpt = get_last_checkpoint(output_dir)
        if last_ckpt:
            print(f"resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, "args2_model.pt"))
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "role2id.json"), "w") as f:
        json.dump({"role2id": role2id, "base_model": base_model}, f)
    return model, tokenizer, lexicon, role2id, id2role


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output-dir", default="outputs/args2_de")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=320)
    p.add_argument("--role-lambda", type=float, default=1.0)
    p.add_argument("--drop-discontinuous", dest="keep_discontinuous",
                   action="store_false", help="train only on contiguous role spans")
    p.set_defaults(keep_discontinuous=True)
    a = p.parse_args()

    model, tokenizer, lexicon, role2id, id2role = train(
        base_model=a.base_model, output_dir=a.output_dir, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, max_length=a.max_length,
        role_lambda=a.role_lambda, keep_discontinuous=a.keep_discontinuous,
    )

    print_report(evaluate_args2_de(
        model, tokenizer, lexicon, role2id, id2role,
        split="test", max_length=a.max_length, keep_discontinuous=a.keep_discontinuous,
    ))
