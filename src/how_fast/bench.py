"""Benchmark engine — orchestrates warmup → baseline → workloads → aggregate → save."""

from __future__ import annotations

import asyncio
import time

from .client import BenchClient
from .config import load_slo_config
from .gpu_metrics import GPUPoller
from .metrics import aggregate_results, check_slos
from .results import save_run
from .schemas import (
    AggregatedMetrics,
    BenchConfig,
    ExperimentConfig,
    RequestResult,
    WorkloadRow,
)
from .warmup import warmup_server
from . import term


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


async def run_experiment(
    experiment: ExperimentConfig,
    server_url: str,
    bench_config: BenchConfig,
    workloads: dict[str, list[WorkloadRow]],
    gpu_monitor_url: str | None = None,
    include_direct: bool = False,
) -> list[AggregatedMetrics]:
    """Run full benchmark for one experiment.

    Steps:
    1. Warm up via gateway (LB → gateway → vLLM)
    2. Start GPU metrics poller (background, polls gpu_monitor.py)
    3. (if --include-direct) Direct baseline for all workloads
    4. For each workload: send n_runs × rows requests via gateway
    5. Stop GPU poller
    6. Aggregate metrics
    7. Check SLOs
    8. Save results
    """
    gateway_client = BenchClient(
        base_url=bench_config.gateway.lb_url,
        model=experiment.vllm.model,
        stream=bench_config.stream,
        timeout_s=bench_config.timeout_per_request_s,
    )

    # Direct client talks to vLLM server directly (no LB/gateway)
    direct_client = BenchClient(
        base_url=server_url.rstrip("/"),
        model=experiment.vllm.model,
        stream=bench_config.stream,
        timeout_s=bench_config.timeout_per_request_s,
    )

    # ── Step 1: Warm up ──
    first_workload = list(workloads.values())[0]
    await warmup_server(
        gateway_client, first_workload, experiment.name, bench_config.warmup
    )

    # ── Step 2: Start GPU poller (/gpu-stats via gpu_monitor.py) ──
    gpu_poller = GPUPoller(
        base_url=gpu_monitor_url or bench_config.gpu_monitor_url,
        experiment=experiment.name,
    )
    gpu_poller.start()

    # ── Step 3: Direct baseline for all workloads (optional) ──
    direct_results: list[RequestResult] = []

    if include_direct:
        direct_exp_name = f"{experiment.name}_direct"

        for workload_name, rows in workloads.items():
            total_direct = len(rows) * bench_config.n_runs
            term.section(f"direct │ {workload_name}  ({len(rows)} × {bench_config.n_runs} = {total_direct} reqs)")
            t_start = time.monotonic()
            for run_i in range(bench_config.n_runs):
                for req_i, row in enumerate(rows):
                    result = await direct_client.send_request(
                        row,
                        direct_exp_name,
                        workload_name,
                        via="direct",
                        temperature=bench_config.temperature,
                    )
                    direct_results.append(result)
                    done = run_i * len(rows) + req_i + 1
                    term.progress(
                        "direct",
                        f"{experiment.name} │ {workload_name}",
                        done, total_direct,
                        suffix=f"run {run_i + 1}/{bench_config.n_runs}",
                    )
            term.progress_done("direct", f"{experiment.name} │ {workload_name}", time.monotonic() - t_start)
    else:
        term.info("bench", term.gray(f"{experiment.name} | direct off — pass --include-direct to enable"))

    # ── Step 4: Benchmark each workload via gateway ──
    all_results: list[RequestResult] = []
    wall_start = time.monotonic()

    for workload_name, rows in workloads.items():
        total_reqs = len(rows) * bench_config.n_runs
        term.section(f"gateway │ {workload_name}  ({len(rows)} × {bench_config.n_runs} = {total_reqs} reqs)")
        t_start = time.monotonic()

        for run_i in range(bench_config.n_runs):
            for req_i, row in enumerate(rows):
                result = await gateway_client.send_request(
                    row,
                    experiment.name,
                    workload_name,
                    via="gateway",
                    temperature=bench_config.temperature,
                )
                all_results.append(result)
                done = run_i * len(rows) + req_i + 1
                term.progress(
                    "gateway",
                    f"{experiment.name} │ {workload_name}",
                    done, total_reqs,
                    suffix=f"run {run_i + 1}/{bench_config.n_runs}",
                )
        term.progress_done("gateway", f"{experiment.name} │ {workload_name}", time.monotonic() - t_start)

    wall_time_s = time.monotonic() - wall_start

    # ── Step 5: Stop GPU poller ──
    gpu_samples = await gpu_poller.stop()

    # ── Step 6: Aggregate gateway + direct results together ──
    combined_results = all_results + direct_results
    aggregated = aggregate_results(
        results=combined_results,
        gpu_samples=gpu_samples,
        hourly_cost_usd=bench_config.gpu.hourly_cost_usd,
        wall_time_s=wall_time_s,
    )

    # ── Step 7: Check SLOs ──
    slo_report: list[dict] = []
    try:
        slo_config = load_slo_config()
        for m in aggregated:
            thresholds = slo_config.workloads.get(m.workload)
            if thresholds is None:
                continue
            violations = check_slos(m, thresholds.model_dump())
            for v in violations:
                slo_report.append({
                    "experiment": m.experiment,
                    "workload": m.workload,
                    "slo": v["slo"],
                    "threshold": v["threshold"],
                    "actual": v["actual"],
                    "pass": False,
                })
            # Also record passed SLOs
            checked_slos = {v["slo"] for v in violations}
            for slo_key, threshold_val in thresholds.model_dump().items():
                if threshold_val is not None and slo_key not in checked_slos:
                    slo_report.append({
                        "experiment": m.experiment,
                        "workload": m.workload,
                        "slo": slo_key,
                        "threshold": threshold_val,
                        "actual": getattr(m, {
                            "max_ttft_p95_s": "ttft_p95_s",
                            "max_total_latency_p95_s": "total_latency_p95_s",
                            "min_tps_p50": "tps_p50",
                            "max_error_rate": "error_rate",
                            "max_cost_per_request_usd": "cost_per_request_usd",
                        }.get(slo_key, ""), None),
                        "pass": True,
                    })
    except FileNotFoundError:
        term.warn("slo", "No slos.yaml found, skipping SLO checks")

    # ── Step 8: Save ──
    run_dir = save_run(
        experiment=experiment,
        request_results=combined_results,
        aggregated=aggregated,
        gpu_samples=gpu_samples if gpu_samples else None,
        baseline_results=direct_results if direct_results else None,
        slo_report=slo_report if slo_report else None,
        wall_time_s=wall_time_s,
    )

    # Print summary
    term.section("results")
    ok_total = sum(m.n_requests - m.n_errors for m in aggregated)
    err_total = sum(m.n_errors for m in aggregated)
    for m in aggregated:
        ttft_str = f"  ttft_p50={m.ttft_p50_s:.3f}s" if m.ttft_p50_s else ""
        status = term.red(f"{m.n_errors} errors") if m.n_errors else term.green("no errors")
        print(
            f"  {term.cyan(m.workload):<30}"
            f"  lat_p50={term.white(f'{m.total_latency_p50_s:.3f}s')}"
            f"{ttft_str}"
            f"  tps_p50={term.white(f'{m.tps_p50:.1f}')}"
            f"  {status}"
        )

    if slo_report:
        passes = sum(1 for s in slo_report if s["pass"])
        fails  = sum(1 for s in slo_report if not s["pass"])
        slo_str = f"{term.green(f'{passes} passed')}  {term.red(f'{fails} failed') if fails else ''}"
        term.info("slo", slo_str)

    term.summary_banner(experiment.name, str(run_dir))
    return aggregated


async def run_all_experiments(
    experiments: list[ExperimentConfig],
    bench_config: BenchConfig,
    workloads: dict[str, list[WorkloadRow]],
    include_direct: bool = False,
) -> None:
    """Run experiments SEQUENTIALLY."""
    server_url = bench_config.server_url
    gpu_monitor_url = bench_config.gpu_monitor_url

    for experiment in experiments:
        term.experiment_banner(
            name=experiment.name,
            model=experiment.vllm.model,
            server=server_url,
            direct=include_direct,
        )

        await run_experiment(
            experiment, server_url, bench_config, workloads,
            gpu_monitor_url=gpu_monitor_url,
            include_direct=include_direct,
        )

    print(f"\n{term.green('All experiments complete.')} Results in: {term.gray('results/')}")
