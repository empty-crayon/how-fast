"""Tests for warmup module — unit tests for health_check logic."""

import pytest

from how_fast.schemas import WarmupConfig


def test_warmup_config_defaults():
    cfg = WarmupConfig()
    assert cfg.health_check_timeout_s == 120
    assert cfg.warmup_requests == 10
    assert cfg.warmup_duration_s == 120
    assert cfg.warmup_max_tokens == 32


def test_warmup_config_override():
    cfg = WarmupConfig(warmup_requests=5, warmup_duration_s=30)
    assert cfg.warmup_requests == 5
    assert cfg.warmup_duration_s == 30
