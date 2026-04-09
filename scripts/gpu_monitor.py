#!/usr/bin/env python3
"""Lightweight nvidia-smi GPU metrics HTTP server — zero external dependencies.

GET /gpu-stats  → {"gpu_util_pct": 85.0, "vram_used_mb": 20480, "vram_total_mb": 24576}
GET /health     → {"status": "ok"}

Usage:
    python gpu_monitor.py [--port 8081]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from http.server import BaseHTTPRequestHandler, HTTPServer

DEFAULT_PORT = 8081


def _query_gpu() -> dict:
    """Shell out to nvidia-smi and return GPU metrics dict."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if result.returncode != 0:
        raise RuntimeError(f"nvidia-smi failed: {result.stderr.strip()}")
    parts = [p.strip() for p in result.stdout.strip().split(",")]
    return {
        "gpu_util_pct": float(parts[0]),
        "vram_used_mb": float(parts[1]),
        "vram_total_mb": float(parts[2]),
    }


class GPUHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path == "/gpu-stats":
            try:
                data = _query_gpu()
                self._json_response(200, data)
            except Exception as e:
                self._json_response(500, {"error": str(e)})
        elif self.path == "/health":
            self._json_response(200, {"status": "ok"})
        else:
            self._json_response(404, {"error": "not found"})

    def _json_response(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        # Silence per-request logs to avoid cluttering terminal
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="GPU metrics HTTP server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    server = HTTPServer((args.host, args.port), GPUHandler)
    print(f"[gpu_monitor] listening on {args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[gpu_monitor] stopped", flush=True)


if __name__ == "__main__":
    main()
