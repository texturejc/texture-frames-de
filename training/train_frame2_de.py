"""
Train the German frame-classification v2 marker-pooled model on SALSA.

The German counterpart of `encoder_parser/train_frame2.py`. Same model
(`FrameMarkerModel`, reused unchanged — it is backbone-agnostic) and same
marker-pooling recipe; only the data source, lexicon, and backbone differ:

  * backbone : deepset/gbert-large      (was microsoft/deberta-v3-large)
  * data     : salsa_frame_data         (was frame2_data / NLTK FrameNet)
  * lexicon  : SalsaLexicon             (was Lexicon)

The end-of-run test eval uses the SALSA candidate mask; if `simplemma` is
installed it is passed as the lemmatizer (candidate coverage 98.4% vs 94.5%
surface-only — see the lexicon measurements).

Run:
    python train_frame2_de.py --epochs 5 --batch-size 16
"""
from __future__ import annotations

import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import argparse
import json
import sys
import time

import numpy as np
import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments

# Reuse the shared, backbone-agnostic model from the English tree (single source
# of truth for the architecture — not vendored, to avoid drift).
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model_frame2 import FrameMarkerModel  # noqa: E402

from salsa_frame_data import (  # noqa: E402
    TRIGGER_END,
    TRIGGER_START,
    build_frame2_dataset,
    find_marker_positions,
    mark_trigger,
)
from salsa_lexicon import SalsaLexicon  # noqa: E402
from salsa_loader import load_frame_examples  # noqa: E402

DEFAULT_BASE_MODEL = "deepset/gbert-large"
DEFAULT_BIASES = [0.0, 2.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]


# --------------------------------------------------------------------------- #
# Collator + metric (identical to the English trainer)                         #
# --------------------------------------------------------------------------- #

class FrameMarkerCollator:
    """Right-pad ids/mask; stack start_pos/end_pos/labels (right-padding keeps
    the marker token indices valid)."""

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, features):
        maxlen = max(len(f["input_ids"]) for f in features)
        input_ids, attn, start, end, labels = [], [], [], [], []
        for f in features:
            pad = maxlen - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_id] * pad)
            attn.append(f["attention_mask"] + [0] * pad)
            start.append(f["start_pos"])
            end.append(f["end_pos"])
            labels.append(f["labels"])
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "start_pos": torch.tensor(start, dtype=torch.long),
            "end_pos": torch.tensor(end, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def make_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        acc = float((np.argmax(logits, axis=-1) == labels).mean())
        return {"accuracy": acc}

    return compute_metrics


# --------------------------------------------------------------------------- #
# German candidate-masked test eval (mirrors eval_frame2, with a lemmatizer)   #
# --------------------------------------------------------------------------- #

def _get_lemmatizer():
    """simplemma German lemmatizer if available, else None (surface-only)."""
    try:
        import simplemma

        return lambda w: simplemma.lemmatize(w, lang="de")
    except Exception:
        print("  [eval] simplemma not installed — candidate mask uses surface "
              "forms only (lower coverage). `pip install simplemma` to enable.")
        return None


@torch.no_grad()
def evaluate_frame2_de(model, tokenizer, lexicon, split="test", max_length=320, biases=None):
    biases = biases if biases is not None else DEFAULT_BIASES
    model.eval()
    device = next(model.parameters()).device
    frame2id = lexicon.frame2id()
    start_id = tokenizer.convert_tokens_to_ids(TRIGGER_START)
    end_id = tokenizer.convert_tokens_to_ids(TRIGGER_END)
    lemmatizer = _get_lemmatizer()

    examples = load_frame_examples(split, drop_unannotated=True)
    correct = {b: 0 for b in biases}
    covered = total = 0
    t0 = time.time()
    for text, trigger_loc, gold_frame in examples:
        enc = tokenizer(
            mark_trigger(text, trigger_loc), truncation=True,
            max_length=max_length, return_tensors="pt",
        )
        sp, ep = find_marker_positions(enc["input_ids"][0].tolist(), start_id, end_id)
        logits = model.encode_logits(
            enc["input_ids"].to(device), enc["attention_mask"].to(device),
            torch.tensor([sp], device=device), torch.tensor([ep], device=device),
        )[0]

        candidates = lexicon.candidate_frames(text, trigger_loc, lemmatizer=lemmatizer)
        cand_ids = [frame2id[c] for c in candidates if c in frame2id]
        gold_id = frame2id.get(gold_frame)
        if gold_frame in candidates:
            covered += 1
        for b in biases:
            masked = logits
            if cand_ids and b:
                masked = logits.clone()
                masked[torch.tensor(cand_ids, device=device)] += b
            if int(masked.argmax()) == gold_id:
                correct[b] += 1
        total += 1
    elapsed = time.time() - t0
    return {
        "by_bias": {b: correct[b] / total if total else 0.0 for b in biases},
        "total": total,
        "lexicon_coverage": covered / total if total else 0.0,
        "ms_per_example": 1000 * elapsed / max(total, 1),
        "split": split,
    }


def print_report(res):
    print(f"\n=== frame2 (German) — split={res['split']}, n={res['total']} ===")
    print(f"candidate coverage: {res['lexicon_coverage']:.3f}   "
          f"{res['ms_per_example']:.1f} ms/example")
    best_b = max(res["by_bias"], key=res["by_bias"].get)
    for b, acc in res["by_bias"].items():
        star = "  <- best" if b == best_b else ""
        print(f"  bias {b:>4}: accuracy {acc:.4f}{star}")


# --------------------------------------------------------------------------- #
# Train                                                                        #
# --------------------------------------------------------------------------- #

def train(
    base_model: str = DEFAULT_BASE_MODEL,
    output_dir: str = "outputs/frame2_de",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-5,
    max_length: int = 320,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    resume: bool = True,
):
    lexicon = SalsaLexicon()
    frame2id = lexicon.frame2id()
    print(f"frame vocabulary: {len(frame2id)} frames (German / SALSA)")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    tokenizer.add_special_tokens({"additional_special_tokens": [TRIGGER_START, TRIGGER_END]})
    start_id = tokenizer.convert_tokens_to_ids(TRIGGER_START)
    end_id = tokenizer.convert_tokens_to_ids(TRIGGER_END)

    model = FrameMarkerModel.from_pretrained(base_model, num_frames=len(frame2id))
    model.resize_token_embeddings(len(tokenizer))

    train_ds = build_frame2_dataset("train", tokenizer, frame2id, start_id, end_id, max_length)
    dev_ds = build_frame2_dataset("dev", tokenizer, frame2id, start_id, end_id, max_length)
    print(f"train examples: {len(train_ds)}   dev examples: {len(dev_ds)}")

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
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,  # keep best + most-recent so a disconnect can resume
        remove_unused_columns=False,  # keep start_pos/end_pos for the model
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        data_collator=FrameMarkerCollator(tokenizer),
        compute_metrics=make_compute_metrics(),
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
    torch.save(model.state_dict(), os.path.join(output_dir, "frame2_model.pt"))
    tokenizer.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "frame2id.json"), "w") as f:
        json.dump({"frame2id": frame2id, "base_model": base_model}, f)
    return model, tokenizer, lexicon


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    p.add_argument("--output-dir", default="outputs/frame2_de")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--max-length", type=int, default=320)
    a = p.parse_args()

    model, tokenizer, lexicon = train(
        base_model=a.base_model, output_dir=a.output_dir, epochs=a.epochs,
        batch_size=a.batch_size, lr=a.lr, max_length=a.max_length,
    )

    print_report(evaluate_frame2_de(model, tokenizer, lexicon, split="test", max_length=a.max_length))
