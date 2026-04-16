"""Deployment script generation — platform-agnostic vLLM + GPU monitor.

Generates a shell script per experiment that starts:
  1. vLLM server on port 8000
  2. gpu_monitor.py on port 8081 (nvidia-smi HTTP server)

Both run as background processes managed by a single script with proper
cleanup on exit (trap). The GPU monitor runs in a separate process and
does NOT interfere with inference — it only shells out to nvidia-smi
on demand when polled by the benchmarking client.
"""

from __future__ import annotations

import shlex
from pathlib import Path

from .config import _project_root
from .schemas import ExperimentConfig, VLLMArgs

SCRIPTS_DIR_NAME = "scripts/experiments"
GPU_MONITOR_PORT = 8081

# gpu_monitor.py embedded as a string constant so it survives pip install.
# The scripts/ directory is not included in the wheel; embedding the source
# here ensures `how-fast generate` works from any installed environment.
_GPU_MONITOR_CODE = '''\
#!/usr/bin/env python3
"""Lightweight nvidia-smi GPU metrics HTTP server — zero external dependencies.

GET /gpu-stats  -> {"gpu_util_pct": 85.0, "vram_used_mb": 20480, "vram_total_mb": 24576}
GET /health     -> {"status": "ok"}

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
        pass  # silence per-request logs


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
        print("\\n[gpu_monitor] stopped", flush=True)


if __name__ == "__main__":
    main()
'''


def _scripts_dir() -> Path:
    d = _project_root() / SCRIPTS_DIR_NAME
    d.mkdir(parents=True, exist_ok=True)
    return d


def _build_vllm_command(args: VLLMArgs) -> list[str]:
    """Convert VLLMArgs to vllm serve CLI command."""
    cmd = [
        "vllm",
        "serve",
        args.model,
        "--revision",
        args.revision,
        "--served-model-name",
        args.model,
        "--host",
        "0.0.0.0",
        "--port",
        "8000",
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--max-num-seqs",
        str(args.max_num_seqs),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.enable_chunked_prefill:
        cmd.append("--enable-chunked-prefill")
    if args.enable_prefix_caching:
        cmd.append("--enable-prefix-caching")
    if args.speculative_config:
        cmd.extend(["--speculative-config", args.speculative_config])

    # Emit any extra fields from the YAML that aren't named fields above.
    # Conversion rules:
    #   key underscores → hyphens  (tensor_parallel_size → --tensor-parallel-size)
    #   hyphenated keys preserved  (reasoning-parser → --reasoning-parser)
    #   bool True  → --flag
    #   bool False → skipped
    #   str/int/float → --flag value
    for key, value in (args.model_extra or {}).items():
        flag = "--" + key.replace("_", "-")
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        elif value is not None:
            cmd.extend([flag, str(value)])

    cmd.extend(args.extra_args)
    return cmd


def generate_script(experiment: ExperimentConfig) -> Path:
    """Generate a platform-agnostic shell script.

    The script starts both vLLM (:8000) and gpu_monitor (:8081) as background
    processes. On SIGINT/SIGTERM (Ctrl-C or kill), both are cleaned up.

    Usage on the GPU server:
        bash scripts/experiments/baseline.sh
    """
    cmd_parts = _build_vllm_command(experiment.vllm)
    vllm_cmd = shlex.join(cmd_parts)

    script = f"""\
#!/usr/bin/env bash
# Auto-generated by how-fast for experiment: {experiment.name}
# {experiment.description}
#
# Starts:
#   vLLM              on port 8000
#   GPU monitor        on port {GPU_MONITOR_PORT}
#
# Usage:
#   bash {experiment.name}.sh
#
# From your laptop, set up SSH tunnels:
#   ssh -L 8000:localhost:8000 -L {GPU_MONITOR_PORT}:localhost:{GPU_MONITOR_PORT} user@gpu-server
#
set -euo pipefail

VLLM_PID=""
GPU_MON_PID=""

cleanup() {{
    echo ""
    echo "[launcher] shutting down..."
    [[ -n "$GPU_MON_PID" ]] && kill "$GPU_MON_PID" 2>/dev/null && echo "[launcher] gpu_monitor stopped"
    [[ -n "$VLLM_PID" ]] && kill "$VLLM_PID" 2>/dev/null && echo "[launcher] vLLM stopped"
    wait 2>/dev/null
    [[ -f "$GPU_MONITOR" ]] && rm -f "$GPU_MONITOR"
    echo "[launcher] done"
}}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  Experiment: {experiment.name}"
echo "  Model:      {experiment.vllm.model}"
echo "  GPU type:   {experiment.gpu_type}"
echo "  vLLM port:  8000"
echo "  GPU monitor: {GPU_MONITOR_PORT}"
echo "============================================================"

# ── Write embedded gpu_monitor to a temp file ──
GPU_MONITOR=$(mktemp /tmp/gpu_monitor_XXXXXX.py)
cat > "$GPU_MONITOR" << 'PYTHON_EOF'
{_GPU_MONITOR_CODE}PYTHON_EOF

# ── Start GPU monitor (lightweight, no deps) ──
echo "[launcher] starting gpu_monitor on :{GPU_MONITOR_PORT}..."
python3 "$GPU_MONITOR" --port {GPU_MONITOR_PORT} &
GPU_MON_PID=$!

# ── Start vLLM ──
echo "[launcher] starting vLLM on :8000..."
{vllm_cmd} &
VLLM_PID=$!

echo "[launcher] waiting for processes (Ctrl-C to stop)..."
wait
"""

    path = _scripts_dir() / f"{experiment.name}.sh"
    path.write_text(script)
    path.chmod(0o755)
    return path


def generate_all(experiments: list[ExperimentConfig]) -> list[Path]:
    """Generate shell scripts for all experiments."""
    paths: list[Path] = []
    for exp in experiments:
        path = generate_script(exp)
        paths.append(path)
    return paths
