#!/usr/bin/env python3
"""Launch the local API, independent training worker, and Vite; Ctrl-C stops all."""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    env = {**os.environ, "PYTHONPATH": str(ROOT / "backend")}
    api_port = str(int(env.get("API_PORT", "8000")))
    web_port = str(int(env.get("WEB_PORT", "5173")))
    env.setdefault("API_PROXY_TARGET", f"http://127.0.0.1:{api_port}")
    origins = [origin.strip() for origin in env.get("CORS_ORIGINS", "").split(",") if origin.strip()]
    origins.extend([f"http://localhost:{web_port}", f"http://127.0.0.1:{web_port}"])
    env["CORS_ORIGINS"] = ",".join(dict.fromkeys(origins))
    subprocess.run([sys.executable, "-m", "alembic", "-c", "backend/alembic.ini", "upgrade", "head"],
                   cwd=ROOT, env=env, check=True)
    commands = [
        ([sys.executable, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", api_port], ROOT),
        ([sys.executable, "-m", "app.worker"], ROOT),
        (["npm", "run", "dev", "--", "--host", "127.0.0.1", "--port", web_port, "--strictPort"], ROOT / "frontend"),
    ]
    children: list[subprocess.Popen] = []
    stopping = False

    def stop(*_):
        nonlocal stopping
        if stopping:
            return
        stopping = True
        for child in children:
            if child.poll() is None:
                child.terminate()

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for index, (command, cwd) in enumerate(commands):
            children.append(subprocess.Popen(command, cwd=cwd, env=env))
            if index == 0:
                deadline = time.monotonic() + 60
                while time.monotonic() < deadline and not stopping:
                    if children[0].poll() is not None:
                        raise SystemExit("API exited during startup. Check its error above.")
                    try:
                        with urllib.request.urlopen(f"http://127.0.0.1:{api_port}/api/health", timeout=1) as response:
                            if response.status == 200:
                                break
                    except (OSError, urllib.error.URLError):
                        time.sleep(0.2)
                else:
                    raise SystemExit("API did not become ready within 60 seconds.")
        print(f"\nNeuroStock: http://localhost:{web_port} • API: http://localhost:{api_port}/docs\n", flush=True)
        while not stopping:
            # A short process wait allows Ctrl-C and early-failure handling.
            for child in children:
                try:
                    code = child.wait(timeout=0.3)
                except subprocess.TimeoutExpired:
                    continue
                if not stopping:
                    print(f"Service exited with code {code}; stopping remaining services.", file=sys.stderr)
                    stop()
                    if code:
                        raise SystemExit(code)
    finally:
        stop()
        for child in children:
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()


if __name__ == "__main__":
    main()
