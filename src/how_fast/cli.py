"""CLI entrypoint for how-fast — generate, verify, bench, single."""

from __future__ import annotations

import argparse
import asyncio
import sys

from .config import (
    load_bench_config,
    load_experiments,
    load_workloads,
)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="how-fast",
        description="Inference benchmarking for NanoServe",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # ── generate ──
    gen_p = sub.add_parser(
        "generate", help="Generate vLLM + GPU monitor launch scripts"
    )
    gen_p.add_argument(
        "--experiment", "-e", nargs="*", help="Generate for specific experiments only"
    )

    # ── verify ──
    verify_p = sub.add_parser("verify", help="Health-check vLLM + GPU monitor")
    verify_p.add_argument(
        "--server-url", help="Override server URL (default from bench.yaml)"
    )
    verify_p.add_argument(
        "--gpu-monitor-url", help="Override GPU monitor URL (default from bench.yaml)"
    )

    # ── bench ──
    bench_p = sub.add_parser("bench", help="Run benchmarks")
    bench_p.add_argument(
        "--experiment", "-e", nargs="*", help="Benchmark specific experiments only"
    )
    bench_p.add_argument(
        "--include-direct", action="store_true", default=False,
        help="Also benchmark vLLM directly (bypass gateway) for comparison",
    )

    # ── single ──
    single_p = sub.add_parser(
        "single", help="Benchmark a single endpoint"
    )
    single_p.add_argument("--endpoint", required=True, help="Base URL of the server")
    single_p.add_argument("--name", required=True, help="Label for this experiment")
    single_p.add_argument("--model", required=True, help="Model name")
    single_p.add_argument(
        "--gpu-monitor-url", help="GPU monitor URL (default from bench.yaml)"
    )

    return parser


def cmd_generate(args: argparse.Namespace) -> None:
    from .deployer import generate_all
    from . import term

    experiments = load_experiments(names=args.experiment)
    term.info("generate", f"generating scripts for {term.white(str(len(experiments)))} experiment(s)...")
    paths = generate_all(experiments)

    for path in paths:
        print(f"  {term.green('✓')} {term.gray(str(path))}")

    print(
        f"\n{term.bold('Copy to your GPU server and run:')}"
        f"\n  {term.cyan(f'bash {paths[0].name}')}"
        f"\n\n{term.bold('Then open SSH tunnels:')}"
        f"\n  {term.cyan('ssh -L 8000:localhost:8000 -L 8081:localhost:8081 user@gpu-server')}"
        f"\n\n{term.bold('Then run:')}"
        f"\n  {term.cyan('how-fast bench')}"
    )


def cmd_verify(args: argparse.Namespace) -> None:
    import httpx
    from . import term

    bench_config = load_bench_config()
    server_url = (args.server_url or bench_config.server_url).rstrip("/")
    gpu_monitor_url = (args.gpu_monitor_url or bench_config.gpu_monitor_url).rstrip("/")

    print(f"\n{term.bold('Checking vLLM server...')}")
    try:
        resp = httpx.get(f"{server_url}/health", timeout=10)
        health_str = term.green("OK") if resp.status_code == 200 else term.red(f"HTTP {resp.status_code}")
    except Exception as e:
        health_str = term.red(f"FAIL ({e})")

    try:
        resp = httpx.get(f"{server_url}/v1/models", timeout=10)
        models_str = term.green("OK") if resp.status_code == 200 else term.red(f"HTTP {resp.status_code}")
    except Exception as e:
        models_str = term.red(f"FAIL ({e})")

    print(f"  {term.dim('health')}  {health_str}")
    print(f"  {term.dim('models')}  {models_str}")

    print(f"\n{term.bold('Checking GPU monitor...')}")
    try:
        resp = httpx.get(f"{gpu_monitor_url}/health", timeout=5)
        gpu_health_str = term.green("OK") if resp.status_code == 200 else term.red(f"HTTP {resp.status_code}")
    except Exception as e:
        gpu_health_str = term.red(f"FAIL ({e})")

    try:
        resp = httpx.get(f"{gpu_monitor_url}/gpu-stats", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            gpu_stats_str = (
                term.green("OK") + term.gray(
                    f"  gpu={data.get('gpu_util_pct', '?')}%  "
                    f"vram={data.get('vram_used_mb', '?')}/{data.get('vram_total_mb', '?')} MB"
                )
            )
        else:
            gpu_stats_str = term.red(f"HTTP {resp.status_code}")
    except Exception as e:
        gpu_stats_str = term.red(f"FAIL ({e})")

    print(f"  {term.dim('health')}  {gpu_health_str}")
    print(f"  {term.dim('stats')}   {gpu_stats_str}")
    print()


def cmd_bench(args: argparse.Namespace) -> None:
    from .bench import run_all_experiments

    bench_config = load_bench_config()
    experiments = load_experiments(names=args.experiment)
    workloads = load_workloads()

    asyncio.run(
        run_all_experiments(
            experiments, bench_config, workloads,
            include_direct=args.include_direct,
        )
    )


def cmd_single(args: argparse.Namespace) -> None:
    from .bench import run_experiment
    from .schemas import ExperimentConfig, VLLMArgs

    bench_config = load_bench_config()
    workloads = load_workloads()

    experiment = ExperimentConfig(
        name=args.name,
        description=f"Manual single-endpoint benchmark: {args.endpoint}",
        vllm=VLLMArgs(model=args.model),
    )

    asyncio.run(
        run_experiment(
            experiment, args.endpoint, bench_config, workloads,
            gpu_monitor_url=args.gpu_monitor_url,
        )
    )


def main() -> None:
    parser = _make_parser()
    args = parser.parse_args()

    commands = {
        "generate": cmd_generate,
        "verify": cmd_verify,
        "bench": cmd_bench,
        "single": cmd_single,
    }

    fn = commands.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
