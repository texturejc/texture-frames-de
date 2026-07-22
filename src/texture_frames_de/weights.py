"""
Load the two trained heads from the Hugging Face Hub (downloaded + cached on
first use). Both are custom classes, so we fetch their `.pt` state_dict + label
map and rebuild — using `AutoModel.from_config` for the backbone so we don't
re-download the base gbert weights (the state_dict already carries them).

The trigger stage is a lexicon rule (see `trigger.py`) and loads no weights.
"""
from __future__ import annotations

import json

import torch


def _load_json(repo: str, filename: str):
    from huggingface_hub import hf_hub_download

    with open(hf_hub_download(repo, filename)) as f:
        return json.load(f)


def _backbone_from_base(base_model: str):
    from transformers import AutoConfig, AutoModel

    cfg = AutoConfig.from_pretrained(base_model)  # tiny config.json download only
    return AutoModel.from_config(cfg)


def load_frame(repo: str, device):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from .model_frame2 import FrameMarkerModel

    tok = AutoTokenizer.from_pretrained(repo)
    meta = _load_json(repo, "frame2id.json")
    frame2id = {k: int(v) for k, v in meta["frame2id"].items()}
    model = FrameMarkerModel(_backbone_from_base(meta["base_model"]), num_frames=len(frame2id))
    model.resize_token_embeddings(len(tok))  # +2 markers, BEFORE load
    state = torch.load(hf_hub_download(repo, "frame2_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tok, frame2id, {v: k for k, v in frame2id.items()}


def load_args(repo: str, device):
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer

    from .model_args2 import Args2Model

    tok = AutoTokenizer.from_pretrained(repo)
    meta = _load_json(repo, "role2id.json")
    role2id = {k: int(v) for k, v in meta["role2id"].items()}
    model = Args2Model(_backbone_from_base(meta["base_model"]), num_roles=len(role2id))
    model.resize_token_embeddings(len(tok))
    state = torch.load(hf_hub_download(repo, "args2_model.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.to(device).eval()
    return model, tok, role2id, {v: k for k, v in role2id.items()}
