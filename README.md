# how-fast

**LLM inference benchmarking tool.** Measures latency, throughput, and GPU utilization across vLLM configurations, with gateway overhead isolation and SLO pass/fail reporting.

Built to answer a specific question: when you're comparing vLLM configurations on a real GPU server behind a real inference gateway, how much of your latency budget is the model, and how much is your infrastructure?

---

## Demo

Running a `prefix_caching` experiment against `unsloth/Phi-3-mini-4k-instruct` on a Lambda A10G.

![demo](docs/demo.jpeg)

---

## Architecture

![System Architecture](docs/how-fast%20sys%20arch.png)

Two request paths run concurrently during a benchmark:

**Gateway path** (always on): `how-fast CLI → SSH tunnel → nginx LB → NanoServe gateway → vLLM :8000`

**Direct path** (`--include-direct`): `how-fast CLI → SSH tunnel → vLLM :8000` — bypasses the entire gateway stack and tags results as `<experiment>_direct`

Diffing the two paths gives you a concrete number: how much latency your observability stack, load balancer, and middleware are consuming at p50 and p99. The p99 delta is usually more interesting than the p50 — gateways that look cheap at median tend to have heavy tails.

GPU metrics come from `gpu_monitor.py`, a stdlib-only HTTP server that shells out to `nvidia-smi`. It's embedded as a heredoc in the generated launch script so nothing extra needs copying to the server.

---

## Workflow

![Benchmark Workflow](docs/how-fast%20workflow.png)

```bash
# 1. Define an experiment
cat config/experiments/chunked_prefill.yaml
```

```yaml
name: chunked_prefill
description: "Chunked prefill enabled"
gpu_type: A10G

vllm:
  model: meta-llama/Meta-Llama-3.1-8B-Instruct
  enable_chunked_prefill: true
  enforce_eager: true
  max_model_len: 4096
```

```bash
# 2. Generate the self-contained launch script
how-fast generate -e chunked_prefill

# 3. Deploy on GPU server (starts vLLM + gpu_monitor sidecar)
scp scripts/experiments/chunked_prefill.sh user@gpu-server:~
ssh user@gpu-server "bash chunked_prefill.sh"

# 4. Open SSH tunnels (keep alive in background)
ssh -L 8000:localhost:8000 -L 8081:localhost:8081 user@gpu-server -N &

# 5. Verify connectivity
how-fast verify

# 6. Run benchmarks
how-fast bench -e chunked_prefill

# 7. Measure gateway overhead on your best config
how-fast bench -e chunked_prefill --include-direct
```

---

## SLO Verification

Define thresholds per workload in `config/slos.yaml`:

```yaml
workloads:
  chat:
    max_ttft_p95_s: 1.0
    max_total_latency_p95_s: 10.0
    min_tps_p50: 20.0
    max_error_rate: 0.01
  classify:
    max_ttft_p95_s: 0.5
    max_total_latency_p95_s: 3.0
```

After each benchmark run, every metric is checked against its threshold. Results are written to `slo_report.json` and visualized as a heatmap and radar chart in the analysis notebook — one row per experiment, one column per SLO, green/red.

The practical workflow is: run without `--include-direct` while iterating across configs to find the best one, then run with `--include-direct` on the winner to produce an honest SLO report that reflects your actual deployed stack.

---

## Output

Each run writes to `results/<experiment>/<timestamp>/`:

| File | Contents |
|------|----------|
| `requests.csv` | Per-request TTFT, total latency, TPS, status, via (gateway/direct) |
| `gpu_metrics.csv` | Utilization and VRAM samples at 2s intervals |
| `summary.json` | p50/p95 aggregates across workloads |
| `slo_report.json` | Pass/fail per metric per workload |
| `meta.json` | Experiment config, wall time, timestamp |

Analysis notebook at `notebooks/analysis.ipynb` renders per-workload comparison tables, GPU utilization overlaid on latency timelines, and the SLO heatmap.

---

## CLI Reference

| Command | Description |
|---------|-------------|
| `how-fast generate` | Generate launch scripts for all experiments |
| `how-fast generate -e <name>` | Generate for a specific experiment |
| `how-fast verify` | Health-check vLLM, gateway, and GPU monitor |
| `how-fast bench` | Benchmark all experiments × all workloads |
| `how-fast bench -e <name>` | Benchmark a specific experiment |
| `how-fast bench -e <name> --include-direct` | + bypass gateway, measure overhead |
| `how-fast single --endpoint URL --name label --model MODEL` | Benchmark any endpoint ad-hoc |

---

## Project Structure

```
how-fast/
├── config/
│   ├── bench.yaml                 # Global settings (URLs, warmup, n_runs)
│   ├── slos.yaml                  # SLO thresholds per workload
│   └── experiments/               # One YAML per vLLM configuration
├── workloads/
│   ├── chat.jsonl
│   ├── classify.jsonl
│   └── synthesize.jsonl
├── src/how_fast/
│   ├── client.py                  # Async HTTP client, TTFT measurement
│   ├── warmup.py                  # Health check + warmup requests
│   ├── metrics.py                 # numpy aggregation + SLO checks
│   ├── gpu_metrics.py             # Background async GPU poller
│   ├── deployer.py                # Launch script generator
│   ├── bench.py                   # Orchestrator
│   └── cli.py
├── notebooks/
│   └── analysis.ipynb
└── results/                       # Benchmark output (gitignored)
```

---

## Setup

```bash
pip install -e ".[dev]"
pytest tests/
```

Requires a remote GPU server running vLLM. The generated launch script handles everything on the server side — vLLM startup and the GPU monitor sidecar are both embedded in the single `.sh` file.