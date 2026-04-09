"""Tests for the async HTTP client."""

import pytest

from how_fast.client import classify_error

import httpx


def test_classify_timeout():
    e = httpx.TimeoutException("read timed out")
    assert classify_error(e) == "timeout"


def test_classify_connect():
    e = httpx.ConnectError("connection refused")
    assert classify_error(e) == "connection"


def test_classify_rate_limit():
    resp = httpx.Response(429, request=httpx.Request("POST", "http://x"))
    e = httpx.HTTPStatusError("rate limited", request=resp.request, response=resp)
    assert classify_error(e) == "rate_limit"


def test_classify_5xx():
    resp = httpx.Response(500, request=httpx.Request("POST", "http://x"))
    e = httpx.HTTPStatusError("server error", request=resp.request, response=resp)
    assert classify_error(e) == "http_5xx"


def test_classify_4xx():
    resp = httpx.Response(400, request=httpx.Request("POST", "http://x"))
    e = httpx.HTTPStatusError("bad request", request=resp.request, response=resp)
    assert classify_error(e) == "http_4xx"


def test_classify_other():
    e = ValueError("something went wrong")
    assert classify_error(e) == "other"
