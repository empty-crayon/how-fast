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


def _add_load_args(parser: argparse.ArgumentParser) -> None:
    """Add --concurrency / --qps / --duration flags (mutually exclusive)."""
    load_group = parser.add_mutually_exclusive_group()
    load_group.add_argument(
        "--concurrency", "-c", type=int, metavar="N",
        help="Override load profile: concurrency mode with N parallel workers",
    )
    load_group.add_argument(
        "--qps", type=float, metavar="RATE",
        help="Override load profile: QPS mode at RATE requests/sec (Poisson)",
    )
    parser.add_argument(
        "--duration", "-d", type=int, metavar="SECS",
        help="Override load profile duration in seconds (default: from bench.yaml)",
    )


def _apply_load_overrides(args: argparse.Namespace, bench_config: "BenchConfig") -> None:
    """Apply CLI --concurrency/--qps/--duration overrides to bench_config."""
    from .schemas import ConcurrencyProfile, QPSProfile

    if getattr(args, "concurrency", None):
        bench_config.load_profile = ConcurrencyProfile(
            concurrent_requests=args.concurrency,
            duration_s=args.duration or bench_config.load_profile.duration_s,
        )
    elif getattr(args, "qps", None):
        bench_config.load_profile = QPSProfile(
            target_qps=args.qps,
            duration_s=args.duration or bench_config.load_profile.duration_s,
        )
    elif getattr(args, "duration", None):
        # Only duration override — keep the existing profile type
        bench_config.load_profile = bench_config.load_profile.model_copy(
            update={"duration_s": args.duration}
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
    verify_p = sub.add_parser("verify", help="Health-check vLLM + GPU monitor (polls until healthy)")
    verify_p.add_argument(
        "--server-url", help="Override server URL (default from bench.yaml)"
    )
    verify_p.add_argument(
        "--gpu-monitor-url", help="Override GPU monitor URL (default from bench.yaml)"
    )
    verify_p.add_argument(
        "--timeout", type=int, default=120, metavar="SECS",
        help="Max time to wait for healthy endpoints (default: 120)",
    )
    verify_p.add_argument(
        "--interval", type=int, default=10, metavar="SECS",
        help="Seconds between polling attempts (default: 10)",
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
    bench_p.add_argument(
        "--one-shot", action="store_true", default=False,
        help="Stop after each prompt has been sent once (no looping over the dataset)",
    )
    _add_load_args(bench_p)

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
    single_p.add_argument(
        "--one-shot", action="store_true", default=False,
        help="Stop after each prompt has been sent once (no looping over the dataset)",
    )
    _add_load_args(single_p)

    # ── sweep ──
    sweep_p = sub.add_parser(
        "sweep",
        help="Run a concurrency or QPS sweep over a set of values for one experiment",
    )
    sweep_p.add_argument(
        "--experiment", "-e", required=True,
        help="Experiment name to sweep (must match a YAML in config/experiments/)",
    )
    sweep_p.add_argument(
        "--include-direct", action="store_true", default=False,
        help="Also benchmark vLLM directly for each step",
    )
    sweep_p.add_argument(
        "--one-shot", action="store_true", default=False,
        help="Stop after each prompt has been sent once (no looping over the dataset)",
    )
    sweep_mode = sweep_p.add_mutually_exclusive_group(required=True)
    sweep_mode.add_argument(
        "--concurrency", "-c", nargs="+", type=int, metavar="N",
        help="Concurrency values to sweep, e.g. --concurrency 1 2 4 8 16",
    )
    sweep_mode.add_argument(
        "--qps", nargs="+", type=float, metavar="RATE",
        help="QPS values to sweep, e.g. --qps 0.5 1 2 4 8",
    )
    sweep_p.add_argument(
        "--duration", "-d", type=int, metavar="SECS",
        help="Duration per step in seconds (default: from bench.yaml)",
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
    import time
    import httpx
    from . import term

    bench_config = load_bench_config()
    server_url = (args.server_url or bench_config.server_url).rstrip("/")
    gpu_monitor_url = (args.gpu_monitor_url or bench_config.gpu_monitor_url).rstrip("/")
    poll_interval = args.interval
    timeout = args.timeout

    term.info("verify", f"Polling endpoints (every {poll_interval}s, timeout {timeout}s)")
    term.info("verify", f"  vLLM:        {term.white(server_url)}")
    term.info("verify", f"  GPU monitor: {term.white(gpu_monitor_url)}")
    print()

    start = time.monotonic()
    attempt = 0

    while True:
        attempt += 1
        elapsed = time.monotonic() - start
        if elapsed > timeout:
            print()
            term.error("verify", f"Timed out after {timeout}s — not all endpoints healthy")
            sys.exit(1)

        term.info("verify", f"attempt {attempt}  ({int(elapsed)}s elapsed)")

        # ── vLLM /health ──
        vllm_ok = False
        try:
            resp = httpx.get(f"{server_url}/health", timeout=5)
            if resp.status_code == 200:
                vllm_ok = True
                term.ok("verify", f"  vLLM /health    {term.green('OK')}")
            else:
                term.warn("verify", f"  vLLM /health    {term.red(f'HTTP {resp.status_code}')}")
        except Exception as e:
            term.warn("verify", f"  vLLM /health    {term.red(f'{type(e).__name__}')}")

        # ── vLLM /v1/models ──
        models_ok = False
        try:
            resp = httpx.get(f"{server_url}/v1/models", timeout=5)
            if resp.status_code == 200:
                models_ok = True
                term.ok("verify", f"  vLLM /v1/models {term.green('OK')}")
            else:
                term.warn("verify", f"  vLLM /v1/models {term.red(f'HTTP {resp.status_code}')}")
        except Exception as e:
            term.warn("verify", f"  vLLM /v1/models {term.red(f'{type(e).__name__}')}")

        # ── GPU monitor /gpu-stats ──
        gpu_ok = False
        try:
            resp = httpx.get(f"{gpu_monitor_url}/gpu-stats", timeout=5)
            if resp.status_code == 200:
                gpu_ok = True
                data = resp.json()
                gpu_info = term.gray(
                    f"gpu={data.get('gpu_util_pct', '?')}%  "
                    f"vram={data.get('vram_used_mb', '?')}/{data.get('vram_total_mb', '?')} MB"
                )
                term.ok("verify", f"  GPU /gpu-stats  {term.green('OK')}  {gpu_info}")
            else:
                term.warn("verify", f"  GPU /gpu-stats  {term.red(f'HTTP {resp.status_code}')}")
        except Exception as e:
            term.warn("verify", f"  GPU /gpu-stats  {term.red(f'{type(e).__name__}')}")

        # ── All healthy? ──
        if vllm_ok and models_ok and gpu_ok:
            print()
            term.ok("verify", f"All endpoints healthy ({int(elapsed)}s, {attempt} attempt{'s' if attempt > 1 else ''})")
            return

        # ── Wait for next attempt ──
        remaining = timeout - (time.monotonic() - start)
        wait = min(poll_interval, max(1, int(remaining)))
        if remaining <= 0:
            continue  # will hit the timeout check at top of loop
        term.info("verify", term.gray(f"  retrying in {wait}s..."))
        print()
        time.sleep(wait)


def cmd_bench(args: argparse.Namespace) -> None:
    from .bench import run_all_experiments

    bench_config = load_bench_config()
    _apply_load_overrides(args, bench_config)
    if getattr(args, "one_shot", False):
        bench_config.exhaust_dataset = True
    experiments = load_experiments(names=args.experiment)
    workloads = load_workloads()

    asyncio.run(
        run_all_experiments(
            experiments, bench_config, workloads,
            include_direct=args.include_direct,
        )
    )


def cmd_sweep(args: argparse.Namespace) -> None:
    from .bench import run_all_experiments
    from .schemas import ConcurrencyProfile, QPSProfile
    from . import term

    bench_config = load_bench_config()
    if getattr(args, "one_shot", False):
        bench_config.exhaust_dataset = True
    experiments = load_experiments(names=[args.experiment])
    workloads = load_workloads()

    base_duration = args.duration or bench_config.load_profile.duration_s

    if args.concurrency:
        steps = [
            ConcurrencyProfile(concurrent_requests=c, duration_s=base_duration)
            for c in args.concurrency
        ]
        sweep_label = f"concurrency sweep: {args.concurrency}  duration={base_duration}s"
    else:
        steps = [
            QPSProfile(target_qps=q, duration_s=base_duration)
            for q in args.qps
        ]
        sweep_label = f"qps sweep: {args.qps}  duration={base_duration}s"

    term.info("sweep", f"{sweep_label}  ({len(steps)} steps)")

    for i, profile in enumerate(steps, 1):
        if isinstance(profile, ConcurrencyProfile):
            step_str = f"conc={profile.concurrent_requests}"
        else:
            step_str = f"qps={profile.target_qps}"
        term.info("sweep", f"step {i}/{len(steps)}  {step_str}  duration={profile.duration_s}s")
        bench_config.load_profile = profile
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
    _apply_load_overrides(args, bench_config)
    if getattr(args, "one_shot", False):
        bench_config.exhaust_dataset = True
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
        "sweep": cmd_sweep,
    }

    fn = commands.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)
    fn(args)


if __name__ == "__main__":
    main()
