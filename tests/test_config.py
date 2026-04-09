"""Tests for config loading."""

from pathlib import Path

from how_fast.config import load_bench_config, load_experiments, load_slo_config, load_workloads

CONFIG_DIR = Path(__file__).parent.parent / "config"
WORKLOADS_DIR = Path(__file__).parent.parent / "workloads"


def test_load_bench_config():
    cfg = load_bench_config(CONFIG_DIR / "bench.yaml")
    assert cfg.gateway.lb_url == "http://localhost:8780"
    assert cfg.gpu.gpu_type == "A10G"
    assert cfg.n_runs == 1
    assert cfg.warmup.warmup_requests == 10


def test_load_experiments():
    exps = load_experiments(CONFIG_DIR / "experiments")
    names = {e.name for e in exps}
    assert "baseline" in names

    baseline = next(e for e in exps if e.name == "baseline")
    assert baseline.vllm.enforce_eager is True
    assert baseline.vllm.enable_chunked_prefill is False


def test_load_experiments_filter():
    exps = load_experiments(CONFIG_DIR / "experiments", names=["baseline"])
    assert len(exps) == 1
    assert exps[0].name == "baseline"


def test_load_slo_config():
    slo = load_slo_config(CONFIG_DIR.parent / "config" / "slos.yaml")
    assert "chat" in slo.workloads
    assert slo.workloads["chat"].max_ttft_p95_s == 1.0


def test_load_workloads():
    wkls = load_workloads(WORKLOADS_DIR)
    assert "chat" in wkls
    assert "classify" in wkls
    assert "synthesize" in wkls
    assert len(wkls["chat"]) == 30
    # Check fields stripped
    row = wkls["chat"][0]
    assert row.request_id.startswith("chat-")
    assert isinstance(row.messages, list)
