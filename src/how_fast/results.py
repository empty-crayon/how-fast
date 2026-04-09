"""Save benchmark results: CSV, JSON, metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .config import get_results_dir
from .schemas import (
    AggregatedMetrics,
    ExperimentConfig,
    GPUSample,
    RequestResult,
)


def _run_dir(experiment: str, results_dir: Path | None = None) -> Path:
    """Create timestamped run directory: results/<experiment>/<timestamp>/"""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
    d = (results_dir or get_results_dir()) / experiment / ts
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_run(
    experiment: ExperimentConfig,
    request_results: list[RequestResult],
    aggregated: list[AggregatedMetrics],
    gpu_samples: list[GPUSample] | None = None,
    baseline_results: list[RequestResult] | None = None,
    slo_report: list[dict] | None = None,
    wall_time_s: float | None = None,
    results_dir: Path | None = None,
) -> Path:
    """Save all outputs for one experiment run. Returns the run directory."""
    run_dir = _run_dir(experiment.name, results_dir)

    # 1. requests.csv — raw request-level data
    if request_results:
        df = pd.DataFrame([r.model_dump() for r in request_results])
        df.to_csv(run_dir / "requests.csv", index=False)

    # 2. summary.json — aggregated per-workload
    if aggregated:
        summary = [m.model_dump() for m in aggregated]
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    # 3. gpu_metrics.csv
    if gpu_samples:
        gpu_df = pd.DataFrame([s.model_dump() for s in gpu_samples])
        gpu_df.to_csv(run_dir / "gpu_metrics.csv", index=False)

    # 4. direct_baseline.json — optional direct-mode results
    if baseline_results:
        baseline = [r.model_dump() for r in baseline_results]
        (run_dir / "direct_baseline.json").write_text(json.dumps(baseline, indent=2))

    # 5. slo_report.json
    if slo_report:
        (run_dir / "slo_report.json").write_text(json.dumps(slo_report, indent=2))

    # 6. meta.json — experiment config + run info
    meta = {
        "experiment": experiment.model_dump(),
        "wall_time_s": wall_time_s,
        "run_dir": str(run_dir),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return run_dir
