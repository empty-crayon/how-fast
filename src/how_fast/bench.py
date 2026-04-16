"""Benchmark engine — orchestrates warmup → baseline → workloads → aggregate → save."""

from __future__ import annotations

import asyncio
import itertools
import random
import time
from collections.abc import Callable

from .client import BenchClient
from .config import load_slo_config, load_warmup_workload
from .gpu_metrics import GPUPoller
from .metrics import aggregate_results, check_slos
from .results import save_run
from .schemas import (
    AggregatedMetrics,
    BenchConfig,
    ConcurrencyProfile,
    ExperimentConfig,
    QPSProfile,
    RequestResult,
    WorkloadRow,
)
from .warmup import warmup_server
from . import term


def _timestamp() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")


# ── Load Engines ────────────────────────────────────────────────────────────


async def _progress_ticker(
    tag: str,
    label: str,
    start_time: float,
    duration: int,
    suffix_fn: Callable[[], str],
    stop_event: asyncio.Event,
    interval: float = 0.25,
    draining_fn: Callable[[], bool] = lambda: False,
    progress_fn: Callable[[], tuple[int, int]] | None = None,
) -> None:
    """Dedicated progress-display coroutine — updates every `interval` seconds
    regardless of what the producer or workers are doing.

    If *progress_fn* is supplied it must return ``(done, total)`` and the bar
    tracks completion count instead of elapsed time (used by exhaust mode).
    """
    while not stop_event.is_set():
        elapsed = time.monotonic() - start_time
        if progress_fn is not None:
            elapsed_i, total_display = progress_fn()
        elif draining_fn():
            elapsed_i = int(elapsed)
            total_display = int(elapsed)
        else:
            elapsed_i = int(min(elapsed, duration))
            total_display = duration
        term.progress(
            tag, label, elapsed_i, total_display,
            suffix=suffix_fn(),
        )
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            pass


async def _run_concurrency(
    client: BenchClient,
    rows: list[WorkloadRow],
    experiment: str,
    workload: str,
    via: str,
    temperature: float,
    profile: ConcurrencyProfile,
    tag: str = "bench",
    exhaust: bool = False,
) -> list[RequestResult]:
    """Closed-loop worker pool: N workers each loop independently for duration_s.

    Each worker: check deadline → pick prompt → send → record result → repeat.
    When duration expires, workers finish their current in-flight request and exit.

    If exhaust=True, each prompt is sent exactly once (no duration limit); the run
    ends when every row has been processed, distributed across all workers.
    """
    results: list[RequestResult] = []
    start_time = time.monotonic()
    label = f"{experiment} │ {workload}"
    n_workers = profile.concurrent_requests
    in_flight = 0

    if exhaust:
        # Pre-fill a queue with every row (sent exactly once)
        queue: asyncio.Queue[WorkloadRow] = asyncio.Queue()
        for i, row in enumerate(rows):
            queue.put_nowait(row.model_copy(update={"request_id": f"{row.request_id}-{i}"}))
        total = len(rows)

        stop_tick = asyncio.Event()
        ticker = asyncio.create_task(_progress_ticker(
            tag, label, start_time, total,
            suffix_fn=lambda: f"active {in_flight}/{min(n_workers, total)}  done {len(results)}/{total}",
            stop_event=stop_tick,
            progress_fn=lambda: (len(results), total),
        ))

        async def worker_exhaust() -> None:
            nonlocal in_flight
            while True:
                try:
                    test_row = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                in_flight += 1
                res = await client.send_request(
                    test_row, experiment, workload, via=via, temperature=temperature
                )
                in_flight -= 1
                results.append(res)

        worker_tasks = [asyncio.create_task(worker_exhaust()) for _ in range(n_workers)]
        await asyncio.gather(*worker_tasks)

    else:
        duration = profile.duration_s
        req_counter = itertools.count()

        stop_tick = asyncio.Event()
        ticker = asyncio.create_task(_progress_ticker(
            tag, label, start_time, duration,
            suffix_fn=lambda: (
                f"active {in_flight}/{n_workers}  done {len(results)}"
                if time.monotonic() - start_time < duration
                else f"finishing {in_flight} in-flight │ done {len(results)}"
            ),
            stop_event=stop_tick,
            draining_fn=lambda: time.monotonic() - start_time >= duration,
        ))

        async def worker() -> None:
            nonlocal in_flight
            while time.monotonic() - start_time < duration:
                idx = next(req_counter)
                row = rows[idx % len(rows)]
                test_row = row.model_copy(
                    update={"request_id": f"{row.request_id}-{idx}"}
                )
                in_flight += 1
                res = await client.send_request(
                    test_row, experiment, workload, via=via, temperature=temperature
                )
                in_flight -= 1
                results.append(res)

        worker_tasks = [asyncio.create_task(worker()) for _ in range(n_workers)]
        await asyncio.gather(*worker_tasks)

    # ── Cleanup ──
    stop_tick.set()
    await ticker

    elapsed = time.monotonic() - start_time
    term.progress_done(tag, label, elapsed)
    return results


async def _run_qps(
    client: BenchClient,
    rows: list[WorkloadRow],
    experiment: str,
    workload: str,
    via: str,
    temperature: float,
    profile: QPSProfile,
    tag: str = "bench",
    exhaust: bool = False,
) -> list[RequestResult]:
    """Open-loop Poisson spawner: fire requests at target_qps, no concurrency cap.

    If exhaust=True, each prompt is fired exactly once at the target QPS rate;
    the run ends when all rows have been dispatched (plus drain of in-flight).
    """
    results: list[RequestResult] = []
    active_tasks: set[asyncio.Task] = set()
    start_time = time.monotonic()
    label = f"{experiment} │ {workload}"
    req_idx = 0
    draining = False
    spawned = 0

    if exhaust:
        total = len(rows)
        stop_tick = asyncio.Event()
        ticker = asyncio.create_task(_progress_ticker(
            tag, label, start_time, total,
            suffix_fn=lambda: (
                f"sent {spawned}/{total}  done {len(results)}/{total}  qps={profile.target_qps}"
                if not draining
                else f"draining {len(active_tasks)} in-flight │ done {len(results)}/{total}"
            ),
            stop_event=stop_tick,
            progress_fn=lambda: (len(results), total),
        ))

        async def fire_exhaust(test_row: WorkloadRow) -> None:
            res = await client.send_request(
                test_row, experiment, workload, via=via, temperature=temperature
            )
            results.append(res)

        for i, row in enumerate(rows):
            test_row = row.model_copy(update={"request_id": f"{row.request_id}-{i}"})
            task = asyncio.create_task(fire_exhaust(test_row))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)
            spawned += 1

            interval = random.expovariate(profile.target_qps)
            await asyncio.sleep(interval)

    else:
        duration = profile.duration_s
        prompt_stream = itertools.cycle(rows)

        stop_tick = asyncio.Event()
        ticker = asyncio.create_task(_progress_ticker(
            tag, label, start_time, duration,
            suffix_fn=lambda: (
                f"sent {spawned}  done {len(results)}  qps={profile.target_qps}"
                if not draining
                else f"finishing {len(active_tasks)} in-flight │ done {len(results)}/{spawned}"
            ),
            stop_event=stop_tick,
            draining_fn=lambda: draining,
        ))

        async def fire(test_row: WorkloadRow) -> None:
            res = await client.send_request(
                test_row, experiment, workload, via=via, temperature=temperature
            )
            results.append(res)

        while time.monotonic() - start_time < duration:
            row = next(prompt_stream)
            test_row = row.model_copy(
                update={"request_id": f"{row.request_id}-{req_idx}"}
            )

            task = asyncio.create_task(fire(test_row))
            active_tasks.add(task)
            task.add_done_callback(active_tasks.discard)
            req_idx += 1
            spawned += 1

            interval = random.expovariate(profile.target_qps)
            deadline = time.monotonic() + interval
            while time.monotonic() < deadline:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                await asyncio.sleep(min(remaining, 0.1))
                if time.monotonic() - start_time >= duration:
                    break

    # ── Drain in-flight with live feedback ──
    draining = True
    if active_tasks:
        await asyncio.gather(*active_tasks, return_exceptions=True)

    # ── Cleanup ──
    stop_tick.set()
    await ticker

    elapsed = time.monotonic() - start_time
    term.progress_done(tag, label, elapsed)
    return results


async def _run_load(
    client: BenchClient,
    rows: list[WorkloadRow],
    experiment: str,
    workload: str,
    via: str,
    temperature: float,
    bench_config: BenchConfig,
    tag: str = "bench",
) -> list[RequestResult]:
    """Dispatch to the correct engine based on load_profile type."""
    profile = bench_config.load_profile
    exhaust = bench_config.exhaust_dataset
    if isinstance(profile, ConcurrencyProfile):
        return await _run_concurrency(
            client, rows, experiment, workload, via, temperature, profile, tag, exhaust
        )
    elif isinstance(profile, QPSProfile):
        return await _run_qps(
            client, rows, experiment, workload, via, temperature, profile, tag, exhaust
        )
    raise ValueError(f"Unknown load profile type: {type(profile)}")


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
    4. For each workload: run load engine via gateway
    5. Stop GPU poller, close clients
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
    warmup_rows = load_warmup_workload()
    await warmup_server(
        gateway_client, warmup_rows, experiment.name, bench_config.warmup
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
        profile = bench_config.load_profile

        for workload_name, rows in workloads.items():
            term.section(f"direct │ {workload_name}  ({profile.type} for {profile.duration_s}s)")
            res = await _run_load(
                direct_client, rows, direct_exp_name, workload_name,
                via="direct", temperature=bench_config.temperature,
                bench_config=bench_config, tag="direct",
            )
            direct_results.extend(res)
    else:
        term.info("bench", term.gray(f"{experiment.name} | direct off — pass --include-direct to enable"))

    # ── Step 3b: Save direct results separately ──
    if include_direct and direct_results:
        direct_aggregated = aggregate_results(
            results=direct_results,
            gpu_samples=None,
            hourly_cost_usd=bench_config.gpu.hourly_cost_usd,
            wall_time_s=None,
        )
        direct_experiment = ExperimentConfig(
            name=f"{experiment.name}_direct",
            description=f"Direct baseline for {experiment.name}",
            vllm=experiment.vllm,
            gpu_type=experiment.gpu_type,
        )
        save_run(
            experiment=direct_experiment,
            request_results=direct_results,
            aggregated=direct_aggregated,
            gpu_samples=None,
            load_profile=bench_config.load_profile,
        )
        term.info("save", f"Direct baseline saved → {direct_experiment.name}/")

        # ── Step 3c: Cooldown — let KV cache evict before gateway run ──
        cooldown_s = 30
        term.info("cooldown", f"Waiting {cooldown_s}s for KV cache eviction before gateway run…")
        await asyncio.sleep(cooldown_s)

    # ── Step 4: Benchmark each workload via gateway ──
    all_results: list[RequestResult] = []
    wall_start = time.monotonic()
    profile = bench_config.load_profile

    for workload_name, rows in workloads.items():
        term.section(f"gateway │ {workload_name}  ({profile.type} for {profile.duration_s}s)")
        res = await _run_load(
            gateway_client, rows, experiment.name, workload_name,
            via="gateway", temperature=bench_config.temperature,
            bench_config=bench_config, tag="gateway",
        )
        all_results.extend(res)

    wall_time_s = time.monotonic() - wall_start

    # ── Step 5: Stop GPU poller, close clients ──
    gpu_samples = await gpu_poller.stop()
    await gateway_client.close()
    await direct_client.close()

    # ── Step 6: Aggregate gateway results (direct saved separately in step 3b) ──
    aggregated = aggregate_results(
        results=all_results,
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
                            "max_total_latency_p50_s": "total_latency_p50_s",
                            "min_tps_p50": "tps_p50",
                            "min_throughput_RPS": "throughput_rps",
                            "max_error_rate": "error_rate",
                            "max_cost_per_request_usd": "cost_per_request_usd",
                        }.get(slo_key, ""), None),
                        "pass": True,
                    })
    except FileNotFoundError:
        term.warn("slo", "No slos.yaml found, skipping SLO checks")

    # ── Step 8: Save gateway results ──
    run_dir = save_run(
        experiment=experiment,
        request_results=all_results,
        aggregated=aggregated,
        gpu_samples=gpu_samples if gpu_samples else None,
        slo_report=slo_report if slo_report else None,
        wall_time_s=wall_time_s,
        load_profile=bench_config.load_profile,
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
