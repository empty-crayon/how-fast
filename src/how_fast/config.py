"""Config loading & validation — YAML + JSONL + experiment discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from .schemas import (
    BenchConfig,
    ExperimentConfig,
    SLOConfig,
    WorkloadRow,
)

# Default paths relative to how-fast root
DEFAULT_BENCH_CONFIG = "config/bench.yaml"
DEFAULT_EXPERIMENTS_DIR = "config/experiments"
DEFAULT_SLO_CONFIG = "config/slos.yaml"
DEFAULT_WORKLOADS_DIR = "workloads"
DEFAULT_RESULTS_DIR = "results"


def _project_root() -> Path:
    """Walk up from this file to find the how-fast root (contains pyproject.toml)."""
    p = Path(__file__).resolve().parent
    for _ in range(5):
        if (p / "pyproject.toml").exists():
            return p
        p = p.parent
    return Path.cwd()


def load_bench_config(path: str | Path | None = None) -> BenchConfig:
    root = _project_root()
    path = Path(path) if path else root / DEFAULT_BENCH_CONFIG
    raw = yaml.safe_load(path.read_text())
    return BenchConfig(**raw)


def load_experiments(
    directory: str | Path | None = None,
    names: list[str] | None = None,
) -> list[ExperimentConfig]:
    root = _project_root()
    d = Path(directory) if directory else root / DEFAULT_EXPERIMENTS_DIR
    experiments: list[ExperimentConfig] = []
    for f in sorted(d.glob("*.yaml")):
        raw = yaml.safe_load(f.read_text())
        exp = ExperimentConfig(**raw)
        if names and exp.name not in names:
            continue
        experiments.append(exp)
    if not experiments:
        raise FileNotFoundError(f"No experiment YAML files found in {d}")
    return experiments


def load_slo_config(path: str | Path | None = None) -> SLOConfig:
    root = _project_root()
    path = Path(path) if path else root / DEFAULT_SLO_CONFIG
    raw = yaml.safe_load(path.read_text())
    return SLOConfig(**raw)


def load_workloads(directory: str | Path | None = None) -> dict[str, list[WorkloadRow]]:
    """Load all .jsonl files from workloads dir. Returns {name: [WorkloadRow, ...]}."""
    root = _project_root()
    d = Path(directory) if directory else root / DEFAULT_WORKLOADS_DIR
    workloads: dict[str, list[WorkloadRow]] = {}
    for f in sorted(d.glob("*.jsonl")):
        name = f.stem  # e.g. "chat" from "chat.jsonl"
        rows: list[WorkloadRow] = []
        for line in f.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rows.append(WorkloadRow(**raw))
        if rows:
            workloads[name] = rows
    if not workloads:
        raise FileNotFoundError(f"No workload .jsonl files found in {d}")
    return workloads



def get_results_dir(path: str | Path | None = None) -> Path:
    root = _project_root()
    d = Path(path) if path else root / DEFAULT_RESULTS_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d
