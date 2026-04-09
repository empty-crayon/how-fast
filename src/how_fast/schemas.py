"""Pydantic models for config, results, and SLOs — single source of truth."""

from __future__ import annotations

from pydantic import BaseModel


# ── Config Models ──


class GatewayConfig(BaseModel):
    """Where the nginx LB lives — all benchmark traffic goes here."""

    lb_url: str  # e.g. "http://localhost:8780"


class GPUConfig(BaseModel):
    gpu_type: str  # e.g. "A10G", "A100"
    hourly_cost_usd: float  # e.g. 1.10 for A10G on Modal


class WarmupConfig(BaseModel):
    health_check_timeout_s: int = 120
    warmup_requests: int = 10
    warmup_duration_s: int = 120  # 2 min floor
    warmup_max_tokens: int = 32


class BenchConfig(BaseModel):
    """Top-level bench.yaml schema."""

    gateway: GatewayConfig
    gpu: GPUConfig
    warmup: WarmupConfig = WarmupConfig()
    server_url: str = "http://localhost:8000"  # vLLM server (via SSH tunnel)
    gpu_monitor_url: str = "http://localhost:8081"  # gpu_monitor.py (via SSH tunnel)
    n_runs: int = 10
    temperature: float = 0.0
    timeout_per_request_s: int = 120
    stream: bool = True


# ── Experiment Models ──


class VLLMArgs(BaseModel):
    """vLLM server arguments. Serialized to Modal deploy command."""

    model: str
    revision: str = "main"
    dtype: str = "float16"
    max_model_len: int = 4096
    max_num_seqs: int = 32
    gpu_memory_utilization: float = 0.90
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    enable_prefix_caching: bool = False
    speculative_config: str | None = None
    extra_args: list[str] = []


class ExperimentConfig(BaseModel):
    """One experiment = one vLLM configuration to benchmark."""

    name: str
    description: str = ""
    vllm: VLLMArgs
    gpu_type: str = "A10G"


# ── Workload Models ──


class WorkloadRow(BaseModel):
    """One row from a workload JSONL file."""

    request_id: str
    messages: list[dict]
    max_tokens: int = 256
    temperature: float | None = None


# ── Result Models ──


class RequestResult(BaseModel):
    """One completed request."""

    experiment: str
    workload: str
    request_id: str
    via: str  # "gateway" or "direct"
    status: str  # "ok" or error class
    error_message: str | None = None
    ttft_s: float | None = None
    total_latency_s: float
    tokens_per_sec: float | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    timestamp_utc: str


class GPUSample(BaseModel):
    """One GPU metrics sample."""

    timestamp_utc: str
    experiment: str
    gpu_util_pct: float
    vram_used_mb: float
    vram_total_mb: float


class AggregatedMetrics(BaseModel):
    """Aggregated stats for one (experiment, workload) pair."""

    experiment: str
    workload: str
    n_requests: int
    n_errors: int
    error_rate: float
    # Latency
    total_latency_mean_s: float
    total_latency_p50_s: float
    total_latency_p95_s: float
    ttft_mean_s: float | None = None
    ttft_p50_s: float | None = None
    ttft_p95_s: float | None = None
    # Throughput
    tps_mean: float
    tps_p50: float
    tps_p95: float
    # GPU
    peak_vram_mb: float | None = None
    mean_vram_mb: float | None = None
    mean_gpu_util_pct: float | None = None
    # Cost
    cost_per_request_usd: float | None = None
    cost_per_token_usd: float | None = None


# ── SLO Models ──


class SLOThresholds(BaseModel):
    """Per-workload SLO thresholds."""

    max_ttft_p95_s: float | None = None
    max_total_latency_p95_s: float | None = None
    min_tps_p50: float | None = None
    max_error_rate: float | None = None
    max_cost_per_request_usd: float | None = None


class SLOConfig(BaseModel):
    """SLO definitions: workload_name → thresholds."""

    workloads: dict[str, SLOThresholds]
