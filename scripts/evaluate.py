#!/usr/bin/env python3
"""Evaluate a registry-approved model against its stored historical test period."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_id", help="ID returned by train.py or GET /api/models")
    parser.add_argument("--output", type=Path, help="Optional output JSON path")
    args = parser.parse_args()
    from app.ml import evaluate_model
    result = json.dumps(evaluate_model(args.model_id), indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(result + "\n")
        print(args.output)
    else:
        print(result)


if __name__ == "__main__":
    main()
