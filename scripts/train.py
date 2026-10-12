#!/usr/bin/env python3
"""Reproducible offline training entry point; never runs in an API request."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="AAPL")
    parser.add_argument("--kind", choices=["lstm", "gru", "all"], default="all")
    parser.add_argument("--lookback", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-samples", type=int, default=1200)
    args = parser.parse_args()
    from app.ml import train_model
    kinds = ["lstm", "gru"] if args.kind == "all" else [args.kind]
    for kind in kinds:
        result = train_model(symbol=args.symbol, kind=kind, lookback=args.lookback,
                             epochs=args.epochs, max_samples=args.max_samples,
                             progress_callback=lambda p: print(json.dumps(p), flush=True))
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
