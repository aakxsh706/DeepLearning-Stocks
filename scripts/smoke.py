#!/usr/bin/env python3
"""Exercise the running real API, saved replay, sensitivity, and event recovery.

Usage: .venv/bin/python scripts/smoke.py [--base-url http://127.0.0.1:8000] [--train]
The optional training check requires the independent worker to be running.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import time

import httpx
from websockets.asyncio.client import connect


async def main(base_url: str, train: bool):
    async with httpx.AsyncClient(base_url=base_url, timeout=60, trust_env=False) as client:
        async def request(method, path, **kwargs):
            response = await client.request(method, path, **kwargs)
            response.raise_for_status()
            return response.json()

        assert (await request("GET", "/api/health"))["status"] == "ok"
        models = await request("GET", "/api/models")
        model = next(m for m in models if m["symbol"] == "AAPL" and m["kind"] == "lstm")
        market = await request("GET", "/api/market/AAPL")
        assert market["mode"] == "Replay" and len(market["candles"]) > 30
        cutoff = market["candles"][-3]["time"]
        forecast = await request("POST", "/api/forecasts", json={"symbol": "AAPL", "model_id": model["id"], "cutoff": cutoff})
        telemetry = forecast["telemetry"]
        assert forecast["lower"] <= forecast["median"] <= forecast["upper"]
        assert forecast["median"] == telemetry["output"]["prices"]["median"]
        assert telemetry["dates"][-1] == forecast["input_cutoff"]
        assert len(telemetry["hidden"]) == model["lookback"]
        assert forecast["actual"] == next(c["close"] for c in market["candles"] if c["time"] == forecast["target_time"])
        assert abs(forecast["base_price"] * (1 + forecast["returns"]["median"]) - forecast["median"]) < 1e-9
        assert all(0 <= v <= 1 for row in telemetry["gates"]["forget"] for v in row)
        replay = await request("GET", f"/api/forecasts/{forecast['id']}/telemetry")
        assert replay == telemetry
        scenario = await request("POST", "/api/what-if", json={"forecast_id": forecast["id"], "feature": "close", "percent": 2})
        assert scenario["sensitivity_only"] and scenario["changes"]
        assert (await request("GET", f"/api/forecasts/{forecast['id']}"))["median"] == forecast["median"]
        invalid = await client.post("/api/what-if", json={"forecast_id": forecast["id"], "feature": "close", "percent": 200})
        assert invalid.status_code == 422
        evaluation = await request("GET", "/api/evaluations", params={"model_id": model["id"]})
        assert evaluation["split"] == "historical_test"
        assert evaluation["metrics"]["count"] == evaluation["baseline"]["count"] == len(evaluation["rows"])
        print("PASS market → trained forecast → stored telemetry → exact replay → what-if → historical evaluation", flush=True)

        ws_base = base_url.replace("http://", "ws://").replace("https://", "wss://")
        async with connect(ws_base + "/ws?after=0", proxy=None) as socket:
            # Read the initial backlog until the just-created forecast event arrives.
            cursor = 0
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), 15))
                if str(event["id"]).isdigit():
                    cursor = max(cursor, int(event["id"]))
                if event["type"] == "inference" and event["payload"].get("forecast_id") == forecast["id"]:
                    break
        second = await request("POST", "/api/forecasts", json={"symbol": "AAPL", "model_id": model["id"], "cutoff": cutoff})
        async with connect(ws_base + f"/ws?after={cursor}", proxy=None) as socket:
            while True:
                event = json.loads(await asyncio.wait_for(socket.recv(), 15))
                if str(event["id"]).isdigit():
                    assert int(event["id"]) > cursor
                    if event["type"] == "inference" and event["payload"].get("forecast_id") == second["id"]:
                        break
        print("PASS WebSocket disconnect → cursor reconnect → missed event recovery", flush=True)

        if train:
            config = {"symbol": "AAPL", "kind": "gru", "lookback": 10, "epochs": 2, "max_samples": 240}
            job = await request("POST", "/api/training", json=config)
            deadline = time.monotonic() + 120
            events = []
            async with connect(ws_base + f"/ws?after={cursor}", proxy=None) as socket:
                while time.monotonic() < deadline:
                    current = await request("GET", "/api/training/" + job["id"])
                    assert (await request("GET", "/api/health"))["status"] == "ok"
                    if current["status"] == "failed":
                        raise AssertionError(current["error"])
                    try:
                        event = json.loads(await asyncio.wait_for(socket.recv(), 1))
                        if event["type"] == "training" and event["payload"].get("id") == job["id"]:
                            events.append(event)
                    except TimeoutError:
                        pass
                    if current["status"] == "completed" and any(e["payload"].get("status") == "completed" for e in events):
                        assert len(current["history"]) == 2 and current["model_id"]
                        assert all(h["gradient_norm"] >= 0 for h in current["history"])
                        print("PASS separate worker training → measured epoch events → checkpoint; API remained responsive", flush=True)
                        break
                else:
                    raise AssertionError("Training worker did not finish within 120 seconds")
        print(json.dumps({"forecast_id": forecast["id"], "model_id": model["id"], "mode": forecast["mode"],
                          "historical_test_mae": evaluation["metrics"]["mae"], "baseline_mae": evaluation["baseline"]["mae"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.base_url.rstrip("/"), args.train))
