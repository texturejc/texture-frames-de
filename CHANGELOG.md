# Changelog

All notable changes to this project will be documented in this file. The format
is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this
project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.1] - 2026-08-06
### Added
- `FrameParser.parse_document(text)`: sentence-splits input with NLTK Punkt
  (German model, `span_tokenize`), calls `parse` per sentence, and shifts
  `trigger_loc` and `Argument.start`/`end` back to offsets in the original
  document. Avoids the cross-sentence context leakage and 320-wordpiece
  truncation that occurred when callers passed multi-sentence text to `parse`.
- Optional `sentencize` extra (`pip install "texture-frames-de[sentencize]"`)
  that pulls in `nltk>=3.8`. `parse_document` raises a helpful `ImportError`
  pointing at the extra when nltk isn't installed, so slim installs stay slim.

### Changed
- `parse`'s docstring now states it expects a single sentence and points at
  `parse_document`.

## [0.2.0]
### Added
- Open-vocabulary mode (`open_vocab=True`): POS-detected content-verb triggers
  outside the SALSA lexicon plus a class-prior correction on the frame head to
  prevent OOV triggers from collapsing to high-frequency generic frames.
