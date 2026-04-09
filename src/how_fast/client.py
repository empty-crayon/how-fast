"""Async OpenAI-compatible HTTP client for benchmarking with TTFT measurement."""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import httpx

from .schemas import RequestResult, WorkloadRow


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def classify_error(e: Exception) -> str:
    """Classify exception into error category."""
    msg = str(e).lower()
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, httpx.ConnectError):
        return "connection"
    if isinstance(e, httpx.HTTPStatusError):
        code = e.response.status_code
        if code == 429:
            return "rate_limit"
        if 400 <= code < 500:
            return "http_4xx"
        if 500 <= code < 600:
            return "http_5xx"
    if "timeout" in msg:
        return "timeout"
    if "connect" in msg:
        return "connection"
    return "other"


class BenchClient:
    """Async OpenAI-compatible client for benchmarking."""

    def __init__(
        self,
        base_url: str,
        model: str,
        stream: bool = True,
        timeout_s: int = 120,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.stream = stream
        self.timeout_s = timeout_s

    async def send_request(
        self,
        row: WorkloadRow,
        experiment: str,
        workload: str,
        via: str = "gateway",
        temperature: float = 0.0,
    ) -> RequestResult:
        """Send one chat completion request and measure latency."""
        url = f"{self.base_url}/v1/chat/completions"
        body = {
            "model": self.model,
            "messages": row.messages,
            "max_tokens": row.max_tokens,
            "temperature": row.temperature if row.temperature is not None else temperature,
            "stream": self.stream,
        }

        start = time.perf_counter()
        ttft_s = None
        completion_tokens = 0
        prompt_tokens = 0

        try:
            async with httpx.AsyncClient(
                timeout=httpx.Timeout(self.timeout_s, connect=30.0)
            ) as client:
                if self.stream:
                    ttft_s, completion_tokens = await self._stream_request(
                        client, url, body, start
                    )
                else:
                    prompt_tokens, completion_tokens = await self._non_stream_request(
                        client, url, body
                    )

            total = time.perf_counter() - start
            tps = completion_tokens / total if total > 0 and completion_tokens > 0 else 0.0

            return RequestResult(
                experiment=experiment,
                workload=workload,
                request_id=row.request_id,
                via=via,
                status="ok",
                ttft_s=ttft_s,
                total_latency_s=round(total, 6),
                tokens_per_sec=round(tps, 2),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                timestamp_utc=_utc_now(),
            )
        except Exception as e:
            total = time.perf_counter() - start
            return RequestResult(
                experiment=experiment,
                workload=workload,
                request_id=row.request_id,
                via=via,
                status=classify_error(e),
                error_message=str(e)[:500],
                total_latency_s=round(total, 6),
                timestamp_utc=_utc_now(),
            )

    async def _stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: dict,
        start: float,
    ) -> tuple[float | None, int]:
        """Stream response, capture TTFT and token count."""
        ttft_s: float | None = None
        tokens = 0
        async with client.stream(
            "POST",
            url,
            json=body,
            headers={"Content-Type": "application/json"},
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[6:]
                if payload.strip() == "[DONE]":
                    break
                try:
                    chunk = json.loads(payload)
                except (json.JSONDecodeError, ValueError):
                    continue
                choices = chunk.get("choices", [])
                if choices and choices[0].get("delta", {}).get("content"):
                    if ttft_s is None:
                        ttft_s = time.perf_counter() - start
                    tokens += 1
        return ttft_s, tokens

    async def _non_stream_request(
        self,
        client: httpx.AsyncClient,
        url: str,
        body: dict,
    ) -> tuple[int, int]:
        """Non-streaming request, return (prompt_tokens, completion_tokens)."""
        resp = await client.post(
            url, json=body, headers={"Content-Type": "application/json"}
        )
        resp.raise_for_status()
        data = resp.json()
        usage = data.get("usage", {})
        return usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
