"""Terminal styling — ANSI colours, progress bars, banners. Zero external deps."""

from __future__ import annotations

import sys

# ── ANSI codes ──────────────────────────────────────────────────────────────
_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_DIM    = "\033[2m"

_GREEN  = "\033[32m"
_CYAN   = "\033[36m"
_YELLOW = "\033[33m"
_RED    = "\033[31m"
_WHITE  = "\033[97m"
_GRAY   = "\033[90m"

# Bright variants
_B_GREEN = "\033[92m"
_B_CYAN  = "\033[96m"
_B_WHITE = "\033[97m"

# ── Helpers ──────────────────────────────────────────────────────────────────

def _tty() -> bool:
    return hasattr(sys.stdout, "isatty") and sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    """Wrap text in an ANSI code only when writing to a real tty."""
    if not _tty():
        return text
    return f"{code}{text}{_RESET}"


def green(t: str)  -> str: return _c(_B_GREEN, t)
def cyan(t: str)   -> str: return _c(_B_CYAN,  t)
def yellow(t: str) -> str: return _c(_YELLOW,  t)
def red(t: str)    -> str: return _c(_RED,      t)
def gray(t: str)   -> str: return _c(_GRAY,     t)
def bold(t: str)   -> str: return _c(_BOLD,     t)
def dim(t: str)    -> str: return _c(_DIM,      t)
def white(t: str)  -> str: return _c(_B_WHITE,  t)


# ── Structured log lines ─────────────────────────────────────────────────────

def info(tag: str, msg: str) -> None:
    """[tag] msg  — cyan tag, normal message."""
    print(f"{cyan(f'[{tag}]')} {msg}", flush=True)


def ok(tag: str, msg: str) -> None:
    """[tag] ✓ msg  — green."""
    print(f"{green(f'[{tag}]')} {green('✓')} {msg}", flush=True)


def warn(tag: str, msg: str) -> None:
    """[tag] ⚠ msg  — yellow."""
    print(f"{yellow(f'[{tag}]')} {yellow('⚠')} {msg}", flush=True)


def error(tag: str, msg: str) -> None:
    """[tag] ✗ msg  — red."""
    print(f"{red(f'[{tag}]')} {red('✗')} {msg}", flush=True)


# ── Progress line (overwrites in place) ─────────────────────────────────────

def progress(tag: str, label: str, done: int, total: int, suffix: str = "") -> None:
    """Overwrite current line with a compact progress bar.

    [tag] label ██████████░░░░░░░░░░  60/100  (suffix)
    """
    bar_width = 20
    filled = int(bar_width * done / total) if total else 0
    bar = green("█" * filled) + dim("░" * (bar_width - filled))
    pct = f"{done}/{total}"
    line = f"\r{cyan(f'[{tag}]')} {white(label)} {bar} {gray(pct)}"
    if suffix:
        line += f"  {dim(suffix)}"
    # Pad to clear any longer previous line
    sys.stdout.write(line.ljust(120) + "  ")
    sys.stdout.flush()


def progress_done(tag: str, label: str, elapsed_s: float) -> None:
    """Print the final 'done' line after a progress sequence."""
    bar = green("█" * 20)
    print(f"\r{cyan(f'[{tag}]')} {white(label)} {bar} {green('done')} {gray(f'({elapsed_s:.1f}s)')}")


# ── Banners ──────────────────────────────────────────────────────────────────

def experiment_banner(name: str, model: str, server: str, direct: bool) -> None:
    w = 62
    line = "─" * w
    mode = "gateway + direct" if direct else "gateway only"
    print()
    print(cyan("┌" + line + "┐"))
    print(cyan("│") + f"  {bold(white('EXPERIMENT'))}  {green(name):<50}" + cyan("│"))
    print(cyan("│") + f"  {dim('model')}   {white(model):<52}" + cyan("│"))
    print(cyan("│") + f"  {dim('server')}  {white(server):<52}" + cyan("│"))
    print(cyan("│") + f"  {dim('mode')}    {yellow(mode):<52}" + cyan("│"))
    print(cyan("└" + line + "┘"))
    print()


def summary_banner(experiment: str, run_dir: str) -> None:
    print()
    print(f"  {green('✓')} {bold(experiment)} {dim('→')} {gray(run_dir)}")


def section(title: str) -> None:
    """Print a dim section divider."""
    print(f"\n{dim('  ── ' + title + ' ' + '─' * max(0, 50 - len(title)))}", flush=True)
