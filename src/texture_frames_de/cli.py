"""Command-line interface: `texture-frames-de "Die Regierung erhöhte die Steuern ."`.

Parses a sentence (positional args, or stdin) and prints the frame annotations,
pretty by default or as JSON with --json.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="texture-frames-de",
        description="German frame-semantic parser (gbert encoder, SALSA).",
    )
    ap.add_argument("text", nargs="*", help="sentence to parse (or pipe via stdin)")
    ap.add_argument("--device", default=None, help="cpu or cuda (default: auto)")
    ap.add_argument("--trigger-threshold", type=float, default=0.5,
                    help="lower to detect more triggers, at some precision cost (default: 0.5)")
    ap.add_argument("--null-bias", type=float, default=2.0,
                    help="raise for fewer/more-precise arguments, lower for higher recall (default: 2.0)")
    ap.add_argument("--json", action="store_true", help="emit JSON instead of pretty text")
    ns = ap.parse_args(argv)

    text = " ".join(ns.text).strip() or sys.stdin.read().strip()
    if not text:
        ap.error("no input text (pass a sentence or pipe one via stdin)")

    from .pipeline import FrameParser  # deferred: avoids torch import for --help

    parser = FrameParser(
        device=ns.device, trigger_threshold=ns.trigger_threshold, null_bias=ns.null_bias
    )
    annotations = parser.parse(text)

    if ns.json:
        print(json.dumps([dataclasses.asdict(a) for a in annotations], indent=2, ensure_ascii=False))
    elif not annotations:
        print("(no frames found)")
    else:
        for a in annotations:
            print(f"[{a.frame}] {a.trigger!r}")
            for arg in a.arguments:
                print(f"    {arg.role:14} {arg.text!r}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
