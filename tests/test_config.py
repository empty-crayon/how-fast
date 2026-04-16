"""Tests for config loading."""

from pathlib import Path

from how_fast.config import load_bench_config, load_experiments, load_slo_config, load_workloads

CONFIG_DIR = Path(__file__).parent.parent / "config"
WORKLOADS_DIR = Path(__file__).parent.parent / "workloads"


def test_load_bench_config():
    cfg = load_bench_config(CONFIG_DIR / "bench.yaml")
    assert cfg.gateway.lb_url == "http://localhost:8780"
    assert cfg.gpu.gpu_type == "A10G"
    assert cfg.load_profile.type == "concurrency"
    assert cfg.load_profile.concurrent_requests == 32
    assert cfg.load_profile.duration_s == 120
    assert cfg.warmup.warmup_requests == 10


def test_load_experiments():
    exps = load_experiments(CONFIG_DIR / "experiments")
    names = {e.name for e in exps}
    assert "Baseline" in names

    baseline = next(e for e in exps if e.name == "Baseline")
    assert baseline.vllm.model == "Qwen/Qwen3.5-4B"
    assert baseline.vllm.max_model_len == 8192


def test_load_experiments_filter():
    exps = load_experiments(CONFIG_DIR / "experiments", names=["Baseline"])
    assert len(exps) == 1
    assert exps[0].name == "Baseline"


def test_load_slo_config():
    slo = load_slo_config(CONFIG_DIR.parent / "config" / "slos.yaml")
    assert "mixed" in slo.workloads
    assert slo.workloads["mixed"].max_ttft_p95_s == 3.0


def test_load_workloads():
    wkls = load_workloads(WORKLOADS_DIR)
    assert "mixed" in wkls
    assert "warmup" not in wkls  # warmup.jsonl is excluded from benchmark workloads
    row = wkls["mixed"][0]
    assert row.request_id.startswith("mixed-")
    assert isinstance(row.messages, list)
