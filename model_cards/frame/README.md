---
license: other
license_name: salsa-tiger-academic
license_link: https://www.coli.uni-saarland.de/projects/salsa/corpus/doc/license.html
language:
- de
library_name: transformers
pipeline_tag: text-classification
tags:
- frame-semantics
- framenet
- salsa
- german
- semantic-parsing
- srl
base_model: deepset/gbert-large
---

# texture-frames-de · frame-classification head

The **frame-classification** stage of
[`texture-frames-de`](https://github.com/texturejc/texture-frames-de), a German
frame-semantic parser. Given a sentence with a marked trigger, it predicts which
of **1,027 frames** the trigger evokes.

It fine-tunes [`deepset/gbert-large`](https://huggingface.co/deepset/gbert-large)
on the **[SALSA](https://www.coli.uni-saarland.de/projects/salsa/) 2.0** corpus
and uses **marker-token pooling**: the trigger is wrapped in entity markers
(`… <t> kündigte </t> …`) and the frame representation is the concatenation of the
two marker tokens' hidden states (not `[CLS]`), focusing the classifier on the
predicate. A single forward pass — no beam search.

> This is one of three stages. Use it through the package rather than alone; the
> pipeline handles trigger detection and argument extraction around it.

## Usage

```bash
pip install git+https://github.com/texturejc/texture-frames-de
```

```python
from texture_frames_de import FrameParser
parser = FrameParser()   # downloads this + the args head on first use
for ann in parser.parse("Die Polizei verhaftete den Verdächtigen am Bahnhof ."):
    print(ann.frame, "|", ann.trigger)
# Arrest | verhaftete
```

At inference the logits are **soft-masked** toward the trigger lemma's candidate
frames (via a bundled SALSA lexicon + `simplemma` lemmatization), so a confident
non-candidate can still win while golds outside the top candidate are recovered.

## Files

| File | What |
| ---- | ---- |
| `frame2_model.pt` | model `state_dict` (backbone + marker-pooling classifier) |
| `frame2id.json` | `{frame name → id}` label map + `base_model` |
| tokenizer files | gbert-large tokenizer with the `<t>` / `</t>` markers added |

The custom head (`FrameMarkerModel`) is defined in the package; loading is handled
by `texture_frames_de.weights.load_frame`.

## Results

Test split (held-out 10% of SALSA sentences), operating point picked on dev:

| Metric | Value |
| ------ | ----- |
| Frame accuracy | **0.9045** (candidate bias 4.0) |
| Candidate-coverage ceiling | 0.984 |
| Speed | ~16 ms/example (single forward pass) |

**Not directly comparable** to the English `texture-frames` frame head — different
corpus, label space (1,027 vs 1,221), and splits. Read as a strong standalone
German result.

## Training

`deepset/gbert-large`, 5 epochs, AdamW lr 1e-5, warmup 0.06, weight decay 0.01,
batch 16, max length 320, bf16. Data: SALSA 2.0, 80/10/10 split by sentence id
(train 30,089 / dev 3,787 / test 3,729 frame instances). See the
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
