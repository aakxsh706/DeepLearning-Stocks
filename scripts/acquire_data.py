#!/usr/bin/env python3
"""Recreate the bundled real daily replay inputs from hash-verified public sources."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from app.data import validate_frame

REVISION = "53cd4f5c9e682f59a9b86b6229aa304e570e09dc"
BASE = f"https://raw.githubusercontent.com/plotly/datasets/{REVISION}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Validate local files and provenance hashes without network")
    args = parser.parse_args()
    data = ROOT / "data"
    provenance = json.loads((data / "provenance.json").read_text())
    if args.verify_only:
        for symbol, info in provenance.items():
            path = data / f"{symbol}.csv"
            if hashlib.sha256(path.read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError(f"{symbol}: bundled data hash mismatch")
            frame = validate_frame(pd.read_csv(path))
            print(f"{symbol}: verified {len(frame)} historical observations ({info['start']} to {info['end']})")
        return
    sources = {}
    for symbol, info in provenance.items():
        url = info["url"]
        if url not in sources:
            with urllib.request.urlopen(url, timeout=90) as response:
                sources[url] = response.read()
        content = sources[url]
        if hashlib.sha256(content).hexdigest() != info["source_sha256"]:
            raise ValueError(f"{symbol}: downloaded source differs from pinned provenance; import aborted")
        frame = pd.read_csv(io.BytesIO(content))
        if symbol == "AAPL":
            frame = frame.rename(columns={"Date": "date", **{f"AAPL.{key.title()}": key for key in ("open", "high", "low", "close", "volume")}})
            frame = frame[["date", "open", "high", "low", "close", "volume"]]
        else:
            frame = frame[(frame.Name == symbol) & (frame.date >= info["start"]) & (frame.date <= info["end"])]
            frame = frame[["date", "open", "high", "low", "close", "volume"]]
        validate_frame(frame)
        text = frame.to_csv(index=False)
        if hashlib.sha256(text.encode()).hexdigest() != info["sha256"]:
            raise ValueError(f"{symbol}: normalized data changed; inspect before replacing the pinned snapshot")
        (data / f"{symbol}.csv").write_text(text)
        print(f"Imported {symbol}: {len(frame)} real observations in Replay mode")


if __name__ == "__main__":
    main()
