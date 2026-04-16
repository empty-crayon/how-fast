"""Health check + warm-up logic with live terminal progress."""

from __future__ import annotations

import asyncio
import time

import httpx

from .client import BenchClient
from .schemas import WarmupConfig, WorkloadRow
from . import term


async def health_check(base_url: str, timeout_s: int = 60) -> bool:
    """Poll /health or /v1/models until 200 or timeout."""
    deadline = time.monotonic() + timeout_s
    # Try both common health endpoints
    endpoints = ["/health", "/v1/models"]
    async with httpx.AsyncClient(timeout=httpx.Timeout(10.0, connect=5.0)) as client:
        while time.monotonic() < deadline:
            for ep in endpoints:
                url = f"{base_url.rstrip('/')}{ep}"
                try:
                    resp = await client.get(url)
                    if resp.status_code == 200:
                        return True
                except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError):
                    pass
            await asyncio.sleep(2)
    return False


async def warmup_server(
    client: BenchClient,
    workload_rows: list[WorkloadRow],
    experiment: str,
    warmup_config: WarmupConfig,
) -> None:
    """Run warm-up requests with live terminal progress.

    Shows:
      [warmup] baseline | health check... OK (2.3s)
      [warmup] baseline | warming up: 5/20 requests (32s / 120s)
      [warmup] baseline | warming up: 20/20 requests (125s / 120s) done
    """
    # Phase 1: Health check
    term.info("warmup", f"{term.white(experiment)} │ health check..." )
    t0 = time.monotonic()
    ok = await health_check(client.base_url, warmup_config.health_check_timeout_s)
    elapsed_hc = time.monotonic() - t0
    if not ok:
        term.error("warmup", f"{experiment} │ server not healthy after {warmup_config.health_check_timeout_s}s")
        raise RuntimeError(
            f"Server at {client.base_url} not healthy after {warmup_config.health_check_timeout_s}s"
        )
    term.ok("warmup", f"{term.white(experiment)} │ healthy {term.gray(f'({elapsed_hc:.1f}s)')}")

    # Phase 2: Warm-up requests (at least warmup_duration_s)
    start = time.monotonic()
    sent = 0
    completed = 0
    target = warmup_config.warmup_requests
    min_duration = warmup_config.warmup_duration_s

    # Build a small warmup workload row with limited tokens
    warmup_rows = []
    for row in workload_rows:
        warmup_rows.append(
            WorkloadRow(
                request_id=f"warmup-{row.request_id}",
                messages=row.messages,
                max_tokens=warmup_config.warmup_max_tokens,
                temperature=0.0,
            )
        )
        if len(warmup_rows) >= target:
            break

    # Dedicated ticker so timer updates smoothly while requests are in-flight
    stop_tick = asyncio.Event()

    async def _tick() -> None:
        while not stop_tick.is_set():
            elapsed = time.monotonic() - start
            term.progress(
                "warmup",
                f"{experiment} │ warming up",
                completed, target,
                suffix=f"{int(elapsed)}s / {min_duration}s",
            )
            try:
                await asyncio.wait_for(stop_tick.wait(), timeout=0.25)
            except asyncio.TimeoutError:
                pass

    ticker = asyncio.create_task(_tick())

    # Hard deadline: cut off even mid-request when duration expires
    deadline = start + min_duration

    while sent < target:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            term.warn("warmup", f"{experiment} │ duration limit ({min_duration}s) reached")
            break

        row = warmup_rows[sent % len(warmup_rows)]
        try:
            await asyncio.wait_for(
                client.send_request(row, experiment, "warmup", via="gateway"),
                timeout=remaining,
            )
            completed += 1
        except asyncio.TimeoutError:
            term.warn("warmup", f"{experiment} │ duration limit ({min_duration}s) reached mid-request")
            break
        sent += 1

    stop_tick.set()
    await ticker
    t1 = time.monotonic() - start
    term.progress_done("warmup", f"{experiment} │ warming up", t1)
