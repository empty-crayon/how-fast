"""numpy-based aggregation: latency stats, TPS, cost model."""

from __future__ import annotations

import numpy as np

from .schemas import AggregatedMetrics, GPUSample, RequestResult


def _safe_percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def aggregate_results(
    results: list[RequestResult],
    gpu_samples: list[GPUSample] | None = None,
    hourly_cost_usd: float | None = None,
    wall_time_s: float | None = None,
) -> list[AggregatedMetrics]:
    """Aggregate per-(experiment, workload) pair."""
    # Group by (experiment, workload)
    groups: dict[tuple[str, str], list[RequestResult]] = {}
    for r in results:
        key = (r.experiment, r.workload)
        groups.setdefault(key, []).append(r)

    # Group GPU samples by experiment
    gpu_by_exp: dict[str, list[GPUSample]] = {}
    if gpu_samples:
        for s in gpu_samples:
            gpu_by_exp.setdefault(s.experiment, []).append(s)

    aggregated: list[AggregatedMetrics] = []
    for (exp, wkl), rlist in groups.items():
        ok = [r for r in rlist if r.status == "ok"]
        n_errors = len(rlist) - len(ok)
        error_rate = n_errors / len(rlist) if rlist else 0.0

        latencies = [r.total_latency_s for r in ok]
        ttfts = [r.ttft_s for r in ok if r.ttft_s is not None]
        tps_vals = [r.tokens_per_sec for r in ok if r.tokens_per_sec]
        total_tokens = sum(r.completion_tokens for r in ok)

        # GPU metrics for this experiment
        exp_gpu = gpu_by_exp.get(exp, [])
        peak_vram = max((s.vram_used_mb for s in exp_gpu), default=None)
        mean_vram = (
            float(np.mean([s.vram_used_mb for s in exp_gpu])) if exp_gpu else None
        )
        mean_gpu_util = (
            float(np.mean([s.gpu_util_pct for s in exp_gpu])) if exp_gpu else None
        )

        # Cost model
        cost_per_req = None
        cost_per_tok = None
        if hourly_cost_usd and wall_time_s and ok:
            total_cost = (wall_time_s / 3600) * hourly_cost_usd
            cost_per_req = total_cost / len(ok)
            if total_tokens > 0:
                cost_per_tok = total_cost / total_tokens

        aggregated.append(
            AggregatedMetrics(
                experiment=exp,
                workload=wkl,
                n_requests=len(rlist),
                n_errors=n_errors,
                error_rate=round(error_rate, 4),
                total_latency_mean_s=round(float(np.mean(latencies)) if latencies else 0.0, 6),
                total_latency_p50_s=round(_safe_percentile(latencies, 50), 6),
                total_latency_p95_s=round(_safe_percentile(latencies, 95), 6),
                ttft_mean_s=round(float(np.mean(ttfts)), 6) if ttfts else None,
                ttft_p50_s=round(_safe_percentile(ttfts, 50), 6) if ttfts else None,
                ttft_p95_s=round(_safe_percentile(ttfts, 95), 6) if ttfts else None,
                tps_mean=round(float(np.mean(tps_vals)) if tps_vals else 0.0, 2),
                tps_p50=round(_safe_percentile(tps_vals, 50), 2),
                tps_p95=round(_safe_percentile(tps_vals, 95), 2),
                peak_vram_mb=round(peak_vram, 1) if peak_vram is not None else None,
                mean_vram_mb=round(mean_vram, 1) if mean_vram is not None else None,
                mean_gpu_util_pct=round(mean_gpu_util, 1) if mean_gpu_util is not None else None,
                cost_per_request_usd=round(cost_per_req, 6) if cost_per_req else None,
                cost_per_token_usd=round(cost_per_tok, 8) if cost_per_tok else None,
            )
        )

    return aggregated


def check_slos(
    metrics: AggregatedMetrics,
    thresholds: dict,
) -> list[dict]:
    """Check SLO thresholds, return list of violations."""
    violations: list[dict] = []
    checks = {
        "max_ttft_p95_s": ("ttft_p95_s", "le"),
        "max_total_latency_p95_s": ("total_latency_p95_s", "le"),
        "min_tps_p50": ("tps_p50", "ge"),
        "max_error_rate": ("error_rate", "le"),
        "max_cost_per_request_usd": ("cost_per_request_usd", "le"),
    }
    for slo_key, (metric_field, op) in checks.items():
        threshold = thresholds.get(slo_key)
        if threshold is None:
            continue
        actual = getattr(metrics, metric_field, None)
        if actual is None:
            continue
        passed = actual <= threshold if op == "le" else actual >= threshold
        if not passed:
            violations.append(
                {
                    "slo": slo_key,
                    "threshold": threshold,
                    "actual": actual,
                    "metric_field": metric_field,
                }
            )
    return violations
