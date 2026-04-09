"""GPU metrics poller — polls /gpu-stats endpoint on the GPU server.

The gpu_monitor.py script (running on the GPU server) exposes
GET /gpu-stats which shells out to nvidia-smi and returns JSON:
    {"gpu_util_pct": 85.0, "vram_used_mb": 20480, "vram_total_mb": 24576}

Accessed from the benchmarking laptop via SSH tunnel.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx

from .schemas import GPUSample


class GPUPoller:
    """Background task that polls the /gpu-stats endpoint for real GPU metrics.

    The endpoint is served by gpu_monitor.py on the GPU server,
    which shells out to nvidia-smi on each request.
    """

    def __init__(
        self,
        base_url: str,
        experiment: str,
        interval_s: float = 2.0,
    ):
        self.gpu_stats_url = base_url.rstrip("/") + "/gpu-stats"
        self.experiment = experiment
        self.interval_s = interval_s
        self.samples: list[GPUSample] = []
        self._task: asyncio.Task | None = None
        self._stop_event = asyncio.Event()

    async def _poll_loop(self) -> None:
        async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
            while not self._stop_event.is_set():
                try:
                    resp = await client.get(self.gpu_stats_url)
                    if resp.status_code == 200:
                        sample = self._parse_response(resp.json())
                        if sample:
                            self.samples.append(sample)
                except (httpx.ConnectError, httpx.TimeoutException, httpx.HTTPStatusError, ValueError):
                    pass  # Server might not be ready yet or nvidia-smi failed
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(), timeout=self.interval_s
                    )
                except asyncio.TimeoutError:
                    pass  # Normal — just means we should poll again

    def _parse_response(self, data: dict) -> GPUSample | None:
        """Parse JSON response from /gpu-stats endpoint."""
        try:
            return GPUSample(
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
                experiment=self.experiment,
                gpu_util_pct=round(float(data["gpu_util_pct"]), 1),
                vram_used_mb=round(float(data["vram_used_mb"]), 1),
                vram_total_mb=round(float(data["vram_total_mb"]), 1),
            )
        except (KeyError, TypeError, ValueError):
            return None

    def start(self) -> None:
        """Start background polling."""
        self._stop_event.clear()
        self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> list[GPUSample]:
        """Stop polling and return collected samples."""
        self._stop_event.set()
        if self._task:
            await self._task
            self._task = None
        return self.samples
