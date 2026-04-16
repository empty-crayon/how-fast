"""Tests for concurrency and QPS load engines."""

import asyncio
import time

import pytest

from how_fast.bench import _run_concurrency, _run_qps
from how_fast.schemas import (
    ConcurrencyProfile,
    QPSProfile,
    RequestResult,
    WorkloadRow,
)


# ── Mock client ──────────────────────────────────────────────────────────────


class MockBenchClient:
    """Fake BenchClient that returns results after a configurable delay."""

    def __init__(self, latency_s: float = 0.01):
        self.latency_s = latency_s
        self.call_count = 0

    async def send_request(
        self,
        row: WorkloadRow,
        experiment: str,
        workload: str,
        via: str = "gateway",
        temperature: float = 0.0,
    ) -> RequestResult:
        self.call_count += 1
        await asyncio.sleep(self.latency_s)
        return RequestResult(
            experiment=experiment,
            workload=workload,
            request_id=row.request_id,
            via=via,
            status="ok",
            ttft_s=self.latency_s * 0.2,
            total_latency_s=self.latency_s,
            tokens_per_sec=100.0,
            prompt_tokens=10,
            completion_tokens=50,
            timestamp_utc="2026-01-01T00:00:00Z",
        )

    async def close(self) -> None:
        pass


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def workload_rows() -> list[WorkloadRow]:
    return [
        WorkloadRow(
            request_id=f"test-{i:03d}",
            messages=[{"role": "user", "content": f"Hello {i}"}],
            max_tokens=32,
        )
        for i in range(5)
    ]


# ── Concurrency engine tests ────────────────────────────────────────────────


def test_concurrency_produces_results(workload_rows):
    """8 workers over 2s with 10ms latency should produce many results."""
    client = MockBenchClient(latency_s=0.01)
    profile = ConcurrencyProfile(concurrent_requests=8, duration_s=2)

    results = asyncio.run(
        _run_concurrency(
            client, workload_rows, "exp", "chat", "gateway", 0.0, profile
        )
    )

    assert len(results) > 50  # 8 workers × 2s / 0.01s ≈ 1600, but overhead
    assert all(r.status == "ok" for r in results)
    assert all(r.experiment == "exp" for r in results)
    assert all(r.workload == "chat" for r in results)


def test_concurrency_single_worker(workload_rows):
    """Single worker should still produce results."""
    client = MockBenchClient(latency_s=0.01)
    profile = ConcurrencyProfile(concurrent_requests=1, duration_s=1)

    results = asyncio.run(
        _run_concurrency(
            client, workload_rows, "exp", "chat", "gateway", 0.0, profile
        )
    )

    assert len(results) > 5
    assert all(r.status == "ok" for r in results)


def test_concurrency_unique_request_ids(workload_rows):
    """Each request should get a unique request_id suffix."""
    client = MockBenchClient(latency_s=0.01)
    profile = ConcurrencyProfile(concurrent_requests=4, duration_s=1)

    results = asyncio.run(
        _run_concurrency(
            client, workload_rows, "exp", "chat", "gateway", 0.0, profile
        )
    )

    ids = [r.request_id for r in results]
    assert len(ids) == len(set(ids)), "Request IDs should be unique"


def test_concurrency_respects_duration(workload_rows):
    """Engine should run for approximately duration_s seconds."""
    client = MockBenchClient(latency_s=0.05)
    profile = ConcurrencyProfile(concurrent_requests=4, duration_s=2)

    start = time.monotonic()
    asyncio.run(
        _run_concurrency(
            client, workload_rows, "exp", "chat", "gateway", 0.0, profile
        )
    )
    elapsed = time.monotonic() - start

    assert elapsed >= 1.8  # should run at least close to 2s
    assert elapsed < 5.0  # shouldn't hang


# ── QPS engine tests ────────────────────────────────────────────────────────


def test_qps_produces_results(workload_rows):
    """10 QPS over 2s should produce ~20 results."""
    client = MockBenchClient(latency_s=0.01)
    profile = QPSProfile(target_qps=10.0, duration_s=2)

    results = asyncio.run(
        _run_qps(client, workload_rows, "exp", "chat", "gateway", 0.0, profile)
    )

    # Poisson variance: expect ~20, tolerate 10-40
    assert 10 <= len(results) <= 40
    assert all(r.status == "ok" for r in results)


def test_qps_low_rate(workload_rows):
    """Low QPS (1.0) over 2s should produce ~2 results."""
    client = MockBenchClient(latency_s=0.01)
    profile = QPSProfile(target_qps=1.0, duration_s=2)

    results = asyncio.run(
        _run_qps(client, workload_rows, "exp", "chat", "gateway", 0.0, profile)
    )

    assert 1 <= len(results) <= 6


def test_qps_drains_inflight(workload_rows):
    """In-flight requests at duration end should still complete (graceful exit)."""
    client = MockBenchClient(latency_s=0.5)  # slow requests
    profile = QPSProfile(target_qps=5.0, duration_s=1)

    results = asyncio.run(
        _run_qps(client, workload_rows, "exp", "chat", "gateway", 0.0, profile)
    )

    # All spawned requests should complete (not be lost)
    assert len(results) >= 3
    assert all(r.status == "ok" for r in results)


def test_qps_unique_request_ids(workload_rows):
    """Each request should get a unique request_id suffix."""
    client = MockBenchClient(latency_s=0.01)
    profile = QPSProfile(target_qps=10.0, duration_s=1)

    results = asyncio.run(
        _run_qps(client, workload_rows, "exp", "chat", "gateway", 0.0, profile)
    )

    ids = [r.request_id for r in results]
    assert len(ids) == len(set(ids)), "Request IDs should be unique"


# ── Schema tests ─────────────────────────────────────────────────────────────


def test_concurrency_profile_defaults():
    p = ConcurrencyProfile(concurrent_requests=8, duration_s=60)
    assert p.type == "concurrency"
    assert p.concurrent_requests == 8
    assert p.duration_s == 60


def test_qps_profile_defaults():
    p = QPSProfile(target_qps=5.0, duration_s=30)
    assert p.type == "qps"
    assert p.target_qps == 5.0
    assert p.duration_s == 30
