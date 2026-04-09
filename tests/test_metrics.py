"""Tests for metrics aggregation."""

from how_fast.metrics import aggregate_results, check_slos
from how_fast.schemas import AggregatedMetrics, GPUSample, RequestResult


def _make_result(**overrides) -> RequestResult:
    result = RequestResult(
        experiment="baseline",
        workload="chat",
        request_id="test-001",
        via="gateway",
        status="ok",
        ttft_s=0.5,
        total_latency_s=2.0,
        tokens_per_sec=25.0,
        prompt_tokens=10,
        completion_tokens=50,
        timestamp_utc="2025-01-01T00:00:00Z",
    )
    return result.model_copy(update=overrides)


def test_aggregate_basic():
    results = [_make_result(total_latency_s=i * 0.5 + 1.0) for i in range(10)]
    agg = aggregate_results(results)
    assert len(agg) == 1
    m = agg[0]
    assert m.experiment == "baseline"
    assert m.workload == "chat"
    assert m.n_requests == 10
    assert m.n_errors == 0
    assert m.total_latency_mean_s > 0


def test_aggregate_with_errors():
    results = [
        _make_result(),
        _make_result(),
        _make_result(status="timeout", ttft_s=None, tokens_per_sec=None),
    ]
    agg = aggregate_results(results)
    assert agg[0].n_errors == 1
    assert agg[0].error_rate > 0.3


def test_aggregate_multiple_workloads():
    results = [
        _make_result(workload="chat"),
        _make_result(workload="classify"),
    ]
    agg = aggregate_results(results)
    assert len(agg) == 2
    workloads = {m.workload for m in agg}
    assert workloads == {"chat", "classify"}


def test_aggregate_with_gpu():
    results = [_make_result()]
    samples = [
        GPUSample(
            timestamp_utc="2025-01-01T00:00:00Z",
            experiment="baseline",
            gpu_util_pct=80.0,
            vram_used_mb=20000.0,
            vram_total_mb=24576.0,
        )
    ]
    agg = aggregate_results(results, gpu_samples=samples)
    assert agg[0].peak_vram_mb == 20000.0
    assert agg[0].mean_gpu_util_pct == 80.0


def test_aggregate_cost_model():
    results = [_make_result() for _ in range(5)]
    agg = aggregate_results(results, hourly_cost_usd=1.10, wall_time_s=60.0)
    m = agg[0]
    assert m.cost_per_request_usd is not None
    assert m.cost_per_request_usd > 0
    assert m.cost_per_token_usd is not None


def test_check_slos_pass():
    m = AggregatedMetrics(
        experiment="baseline",
        workload="chat",
        n_requests=10,
        n_errors=0,
        error_rate=0.0,
        total_latency_mean_s=2.0,
        total_latency_p50_s=2.0,
        total_latency_p95_s=3.0,
        ttft_p50_s=0.3,
        ttft_p95_s=0.5,
        tps_mean=30.0,
        tps_p50=30.0,
        tps_p95=35.0,
    )
    violations = check_slos(m, {"max_ttft_p95_s": 1.0, "min_tps_p50": 20.0})
    assert len(violations) == 0


def test_check_slos_violation():
    m = AggregatedMetrics(
        experiment="baseline",
        workload="chat",
        n_requests=10,
        n_errors=0,
        error_rate=0.0,
        total_latency_mean_s=2.0,
        total_latency_p50_s=2.0,
        total_latency_p95_s=3.0,
        ttft_p50_s=0.3,
        ttft_p95_s=1.5,  # Exceeds threshold
        tps_mean=30.0,
        tps_p50=30.0,
        tps_p95=35.0,
    )
    violations = check_slos(m, {"max_ttft_p95_s": 1.0})
    assert len(violations) == 1
    assert violations[0]["slo"] == "max_ttft_p95_s"
