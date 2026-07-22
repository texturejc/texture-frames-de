---
license: other
license_name: salsa-tiger-academic
license_link: https://www.coli.uni-saarland.de/projects/salsa/corpus/doc/license.html
language:
- de
library_name: transformers
pipeline_tag: token-classification
tags:
- frame-semantics
- framenet
- salsa
- german
- semantic-parsing
- srl
- argument-extraction
base_model: deepset/gbert-large
---

# texture-frames-de · argument-extraction head

The **argument-extraction** stage of
[`texture-frames-de`](https://github.com/texturejc/texture-frames-de), a German
frame-semantic parser. Given a sentence with a marked trigger and its frame, it
finds the spans that fill the frame's roles (frame elements) and labels each.

It fine-tunes [`deepset/gbert-large`](https://huggingface.co/deepset/gbert-large)
on the **[SALSA](https://www.coli.uni-saarland.de/projects/salsa/) 2.0** corpus
with a **detect-then-classify** design — two heads on one backbone, a single
forward pass:

- **Head A — span detection:** a role-agnostic 3-class BIO tagger (`O`/`B`/`I`),
  "is this token part of *an* argument?". Dense signal, arbitrary-length spans.
- **Head B — role classification:** for each detected span, pool its tokens
  (`start ⊕ end ⊕ mean`) and classify into **only the current frame's frame
  elements** (plus a `NULL` reject class), masked via the bundled lexicon.

The input carries the predicate marker and the frame's FE menu
(`{frame} [FE1; FE2; …] : … <t> {trigger} </t> …`). A **`NULL`-bias** at inference
sets the precision/recall operating point.

> This is one of three stages. Use it through the package rather than alone.

## Usage

```bash
pip install git+https://github.com/texturejc/texture-frames-de
```

```python
from texture_frames_de import FrameParser
parser = FrameParser()   # downloads this + the frame head on first use
for ann in parser.parse("Die Polizei verhaftete den Verdächtigen ."):
    print([(a.role, a.text) for a in ann.arguments])
# [('Authorities', 'Die Polizei'), ('Suspect', 'den Verdächtigen')]
```

Lower `null_bias` (default 2.0) for higher argument recall:
`FrameParser(null_bias=0.0)`.

## Files

| File | What |
| ---- | ---- |
| `args2_model.pt` | model `state_dict` (backbone + detection + role heads) |
| `role2id.json` | `{role name → id}` label map (incl. `<NULL>`) + `base_model` |
| tokenizer files | gbert-large tokenizer with the `<t>` / `</t>` markers added |

The custom head (`Args2Model`) is defined in the package; loading is handled by
`texture_frames_de.weights.load_args`.

## Results

Test split (held-out 10% of SALSA sentences), operating point picked on dev:

| Metric | Value |
| ------ | ----- |
| Weighted F1 (non-core FEs = 0.5) | **0.844** (P 0.884 / R 0.808, NULL-bias 2.0) |
| Speed | ~17 ms/example (single forward pass) |

**Not directly comparable** to the English `texture-frames` args head: SALSA role
spans are syntactic *constituents* (clean boundaries), which flatters exact-span
F1 relative to FrameNet's looser character spans. Discontinuous role spans (13.8%
of gold, from German verb brackets / extraposition) are represented and scored as
their enclosing span. Read as a strong standalone German result.

## Training

`deepset/gbert-large`, 5 epochs, AdamW lr 1e-5, warmup 0.06, weight decay 0.01,
batch 16, max length 320, bf16, 4 sampled `NULL` negative spans/example. Data:
SALSA 2.0, 80/10/10 split by sentence id. See the
[repo](https://github.com/texturejc/texture-frames-de) for the training notebook.

## Licence

**Code (the package): MIT.** **Weights: for non-commercial research use.** They are
trained on **SALSA**, layered on **TIGER** — both **academic / non-commercial**
licences, with SALSA additionally restricting commercial use of derived data.
Review the [SALSA](https://www.coli.uni-saarland.de/projects/salsa/corpus/) and
TIGER licence terms before any commercial use or redistribution. The corpus itself
is not distributed here and must be obtained under licence.

## Citation

```bibtex
@software{texture_frames_de,
  author = {Carney, James},
  title  = {texture-frames-de: a German frame-semantic parser (gbert / SALSA)},
  url    = {https://github.com/texturejc/texture-frames-de},
  year   = {2026}
}
```

Builds on David Chanin's `frame-semantic-transformer` and its encoder
rearchitecture [`texture-frames`](https://github.com/texturejc/Texture_Frames);
thanks to the SALSA and TIGER projects and to deepset for `gbert-large`.
