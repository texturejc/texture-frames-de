# texture-frames-de

**A fast German frame-semantic parser** — the German counterpart of
[`texture-frames`](https://github.com/texturejc/Texture_Frames). It fine-tunes
[`deepset/gbert-large`](https://huggingface.co/deepset/gbert-large) on the
**[SALSA](https://www.coli.uni-saarland.de/projects/salsa/) corpus** and reuses
the same encoder architecture: single-forward-pass task heads (no beam search),
marker-token pooling for frames, and a detect-then-classify head for arguments.

```python
from texture_frames_de import FrameParser
parser = FrameParser()
for ann in parser.parse("Die Polizei verhaftete den Verdächtigen am Bahnhof ."):
    print(ann.frame, "|", ann.trigger, "|", [(a.role, a.text) for a in ann.arguments])
# Arrest | verhaftete | [('Authorities', 'Die Polizei'), ('Suspect', 'den Verdächtigen')]
```

---

## Contents

1. [Background: semantic frames & SALSA](#background-semantic-frames--salsa)
2. [Results](#results)
3. [Coverage: what it can and can't parse](#coverage-what-it-can-and-cant-parse)
4. [Installation](#installation)
5. [Usage](#usage)
6. [How it works](#how-it-works)
7. [Training](#training)
8. [Model checkpoints](#model-checkpoints)
9. [Acknowledgements](#acknowledgements)
10. [Citation & license](#citation--license)

---

## Background: semantic frames & SALSA

A **semantic frame** is a schematic representation of a situation together with
its participants. The verb *verhaften* (to arrest) evokes an **Arrest** frame,
with roles (**frame elements**) like **Authorities** and **Suspect**. Recognising
the frame and filling its roles turns a flat sentence into *who did what to whom*.

**[SALSA](https://www.coli.uni-saarland.de/projects/salsa/)** (Burchardt et al.,
LREC 2006) is the German counterpart of Berkeley FrameNet: it adds a
role-semantic layer to the syntactically-annotated **TIGER** newspaper corpus,
using FrameNet 1.2/1.3 frames where they apply and **lemma-specific "proto-frames"**
elsewhere. This parser is trained on **SALSA Release 2.0** (~24k sentences,
~37.6k frame instances, ~66k role labels, 1,027 frames). Frame-semantic parsing
is conventionally split into three steps, performed here in order:

| Step | Question | Example |
| ---- | -------- | ------- |
| **1. Trigger identification** | which words evoke a frame? | *verhaftete* |
| **2. Frame classification** | which frame does each trigger evoke? | *verhaftete* → **Arrest** |
| **3. Argument extraction** | which spans fill the roles? | Authorities = *Die Polizei* |

---

## Results

Evaluated on a held-out test split (10% of SALSA sentences, split by sentence id).
Operating points are **picked on dev, reported on test**.

| Task | Metric | Result |
| ---- | ------ | ------ |
| Trigger identification | word-level F1 | **0.876** (P 0.892 / R 0.861) |
| Frame classification | accuracy | **0.9045** (candidate bias 4.0; coverage ceiling 0.984) |
| Argument extraction | weighted F1 (non-core = 0.5) | **0.844** (P 0.884 / R 0.808; NULL-bias 2.0) |
| Inference | single forward pass / stage | ~16–17 ms/example |

**Read these honestly:**

- **These numbers are *not* directly comparable to the English `texture-frames`.**
  Different corpus (SALSA vs FrameNet), label space (1,027 vs 1,221 frames),
  splits, and — importantly — **span conventions**: SALSA role spans are
  syntactic *constituents* (clean boundaries), which flatters exact-span F1
  relative to FrameNet's looser character spans. Read them as strong *standalone*
  German numbers, not as "beating" the English parser.
- **Trigger identification is a lexicon rule, not a neural model** (see
  [How it works](#how-it-works)) — SALSA's partial annotation makes a learned
  closed-world tagger unsound. The rule's F1 (0.876) is high *within its coverage*;
  its real limit is **coverage**, quantified below.
- **Frame classification** reaches 91.9% of its candidate-coverage ceiling (0.984);
  the residual ~8 points is discrimination among valid candidates.
- **Discontinuous role spans** (13.8% of gold, from German verb brackets /
  extraposition) are represented and scored as their enclosing span on both sides.

---

## Coverage: what it can and can't parse

SALSA annotated only ~685 target *lemmas*, so the parser is a **partial** parser —
accurate on covered predicates, silent on the rest. Measured over content words in
TIGER (its target domain):

| POS | fires on (token, frequency-weighted) | vocabulary covered (types) |
| --- | --- | --- |
| Full verbs | **49.6%** | 13.0% (474 / 3,644 lemmas) |
| Common nouns | 15.9% | 1.0% (237 / 24,828) |
| Proper nouns | ~0% (correctly never fire) | — |

SALSA targeted *high-frequency* lemmas, so its ~665 covered content lemmas cover
about **half of verb occurrences** despite being only 13% of verb types. The long
tail is silent: e.g. *ernennen* ("to appoint") was never annotated in SALSA, so a
sentence whose only predicate is *ernennen* returns no annotations. This is the
honest ceiling of any SALSA-trained parser, not an implementation limit. (The token
figures are a mild over-estimate — SALSA sentences are enriched for covered lemmas.)

---

## Installation

Requires **Python ≥ 3.9**. A GPU is optional (CPU works, slower).

```bash
pip install git+https://github.com/texturejc/texture-frames-de
```

This pulls in `torch`, `transformers`, `sentencepiece`, `huggingface_hub`,
`simplemma`, and `numpy`. On **first use**, the two model checkpoints download
from the Hugging Face Hub (~1.3 GB each) and are cached. The lexicon and trigger
data are bundled with the package (328 KB) — no corpus download needed.

```python
from texture_frames_de import FrameParser
parser = FrameParser()                 # first call downloads + caches the weights
print(parser.parse("Der Chef gab dem Kunden das Essen ."))
```

`simplemma` handles German inflection for the trigger rule and frame candidate
mask; without it the parser still runs but with lower recall.

---

## Usage

`FrameParser.parse(text)` returns a `list[FrameAnnotation]`, one per detected trigger:

```python
@dataclass
class FrameAnnotation:
    trigger: str
    trigger_loc: int      # char offset in the sentence
    frame: str
    arguments: list       # list[Argument]

@dataclass
class Argument:
    role: str
    text: str
    start: int            # char offset (-1 if not locatable)
    end: int
```

### Basic

```python
from texture_frames_de import FrameParser
parser = FrameParser()

for ann in parser.parse("Die Regierung kündigte an , die Steuern zu erhöhen ."):
    print(f"[{ann.frame}] trigger={ann.trigger!r}")
    for a in ann.arguments:
        print(f"    {a.role:14} {a.text!r}")
# [Heralding] trigger='kündigte'
#     Communicator   'Die Regierung'
#     Event          'die Steuern zu erhöhen'
# [Cause_change_of_scalar_position] trigger='erhöhen'
#     Agent          'Die Regierung'
#     Item           'die Steuern'
```

> **Tokenization note.** Like SALSA/TIGER, the parser treats whitespace-separated
> tokens as words, so **separate punctuation with spaces** (`… erhöhen .`) for best
> alignment.

### Choosing device and operating points

```python
parser = FrameParser(
    device="cuda",           # "cpu" or "cuda"; defaults to cuda if available
    frame_bias=4.0,          # candidate soft-mask strength (dev-picked)
    null_bias=2.0,           # NULL-reject threshold for args (↑ = higher precision)
    trigger_threshold=0.5,   # ↓ detects more triggers, at some precision cost
)
```

Lower `null_bias` for higher argument recall (recovers peripheral roles like
*Place*/*Time* that were detected but rejected); lower `trigger_threshold` to fire
on more predicates.

### Command line

```bash
texture-frames-de "Die Polizei verhaftete den Verdächtigen ."
texture-frames-de --json "Der Chef gab dem Kunden das Essen ."
echo "Sie gewann das Rennen ." | texture-frames-de
```

### JSON

```python
import dataclasses, json
out = [dataclasses.asdict(a) for a in parser.parse("Sie verkaufte ihr Fahrrad .")]
print(json.dumps(out, indent=2, ensure_ascii=False))
```

---

## How it works

All heads condition on the trigger in context; the design goal is **encoder + task
heads** rather than sequence-to-sequence generation — a single forward pass per
stage, no beam search.

### 1. Trigger identification — a lexicon rule (not a neural head)

SALSA is *partially annotated*: it selected ~685 target lemmas and annotated their
occurrences, so a sentence's unmarked words are **not reliable negatives** (83% of
covered-lemma occurrences are unannotated). Training a closed-world token tagger on
those would poison it. But SALSA's closed-lemma design makes trigger detection a
**lexicon-membership** problem: a token fires iff its lemma's *annotation rate*
(fraction of corpus occurrences that were annotated as a target) exceeds a
threshold. Built on train, evaluated on test with `simplemma` lemmatization, this
scores **F1 0.876** — above what a poisoned neural tagger could achieve.

### 2. Frame classification — marker-token pooling

A sequence classifier over the 1,027 frames. The trigger is wrapped in entity
markers (`… <t> kündigte </t> …`) and the frame representation is the concatenation
of the **`<t>` / `</t>` marker hidden states** (not `[CLS]`), focusing the
classifier on the predicate. At inference, logits are **soft-masked** toward the
lexicon's candidate frames for the trigger's lemma (via `simplemma`), recovering
golds while letting a confident non-candidate still win. Candidate coverage on test
is **0.984**.

For multiword targets (separable verbs like *ankündigen* → *kündigte … an*, or
particle+verb like *zu erhöhen*), the trigger is anchored on the **content-head
token** (using TIGER POS), not the leftmost token — so the model marks the verb,
never a preposition or particle.

### 3. Argument extraction — detect-then-classify

Two heads on one backbone:

- **Head A — span detection:** a 3-class BIO tagger (`O`/`B`/`I`), role-agnostic —
  "is this token part of *an* argument?". Dense signal, arbitrary-length spans.
- **Head B — role classification:** for each detected span, pool its tokens
  (`start ⊕ end ⊕ mean`) and classify into **only the current frame's frame
  elements** (plus a `NULL` reject class), masked via the lexicon.

The input carries the predicate marker and the frame's FE menu
(`{frame} [FE1; FE2; …] : … <t> {trigger} </t> …`). Training adds sampled `NULL`
negative spans so Head B learns to reject Head A's spurious detections; at inference
a **`NULL`-bias** picks the precision/recall operating point.

---

## Training

| Setting | Value |
| ------- | ----- |
| Backbone | `deepset/gbert-large` |
| Data | SALSA Release 2.0 (SALSA/TIGER XML) |
| Split | 80/10/10 by sentence id (train 30,089 / dev 3,787 / test 3,729 frame instances) |
| Precision | bf16 mixed (weights fp32) |
| Optimiser | AdamW, lr 1e-5, warmup 0.06, weight decay 0.01 |
| Batch / length | 16 / 320 |
| Epochs | 5 |
| Hardware | Google Colab (A100 / L4) |

Training and inference notebooks are in the repo (`notebooks/`): `train_frame2_de`,
`train_args2_de`, `parse_de`, and `tune_args_de`. They checkpoint to Google Drive
and auto-resume, so a Colab disconnect is recoverable. **The SALSA corpus is not
distributed here** — obtain it under licence from the SALSA project and point the
notebooks at it.

---

## Model checkpoints

Public on the Hugging Face Hub; downloaded and cached automatically:

- Frame — [`texturejc/texture-frames-de-frame`](https://huggingface.co/texturejc/texture-frames-de-frame)
- Arguments — [`texturejc/texture-frames-de-args`](https://huggingface.co/texturejc/texture-frames-de-args)

(The trigger stage is a bundled lexicon rule and has no checkpoint.)

---

## Acknowledgements

This work builds directly on **David Chanin's**
[`frame-semantic-transformer`](https://github.com/chanind/frame-semantic-transformer)
and its encoder rearchitecture in
[`texture-frames`](https://github.com/texturejc/Texture_Frames), which shaped the
three-stage decomposition, marker pooling, and detect-then-classify design. We
thank the **SALSA** project (Aljoscha Burchardt, Katrin Erk, Anette Frank, Andrea
Kowalski, Sebastian Padó, Manfred Pinkal) and the **TIGER** project for the corpus,
**deepset** for `gbert-large`, and the **simplemma** authors.

---

## Citation & license

```bibtex
@software{texture_frames_de,
  author = {Carney, James},
  title  = {texture-frames-de: a German frame-semantic parser (gbert / SALSA)},
  url    = {https://github.com/texturejc/texture-frames-de},
  year   = {2026}
}
```

**Code license: MIT.**

**⚠️ Data & model-weight terms — read before redistributing.** The models are
trained on the **SALSA** corpus, which is layered on **TIGER**. Both carry
**academic / non-commercial** licences, and SALSA additionally restricts commercial
use of derived data. The corpus itself is **not** included in this repository and
must be obtained under licence from the
[SALSA project](https://www.coli.uni-saarland.de/projects/salsa/corpus/). The
published model weights are provided for **non-commercial research use**; review
the SALSA and TIGER licence terms before any commercial use or redistribution.

- SALSA: Burchardt, Erk, Frank, Kowalski, Padó & Pinkal (2006), *The SALSA Corpus:
  a German Corpus Resource for Lexical Semantics*, LREC.
- TIGER: Brants et al. (2004), *TIGER: Linguistic Interpretation of a German Corpus*.
