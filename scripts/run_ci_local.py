#!/usr/bin/env python3
"""Local quality gate — see docs/CI_LOCAL.md."""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

CI_DOC = "docs/CI_LOCAL.md"
CI_ENV = {
    "PIP_DISABLE_PIP_VERSION_CHECK": "1",
    "PYTHONUTF8": "1",
    "PYTHONIOENCODING": "utf-8",
}

ALL_JOBS = ('security')
FAST_JOBS = ('security')


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent

def _ci_env() -> dict[str, str]:
    env = dict(os.environ)
    env.update(CI_ENV)
    return env

def _banner(title: str) -> None:
    width = max(len(title) + 4, 72)
    print(); print("=" * width); print(f"  {title}"); print("=" * width)

def _step(title: str, cmd: Sequence[str], *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n--- {title} ---")
    print("$", " ".join(cmd))
    subprocess.run(list(cmd), cwd=str(cwd), env=env, check=True)

def _shell(title: str, script: str, *, cwd: Path, env: dict[str, str]) -> None:
    print(f"\n--- {title} ---")
    subprocess.run(["bash", "-c", script], cwd=str(cwd), env=env, check=True)

def _python() -> str:
    return os.environ.get("PYTHON_BIN") or shutil.which("python3") or sys.executable

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run the local quality gate (docs/CI_LOCAL.md).")
    parser.add_argument("--job", choices=ALL_JOBS, action="append")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args(argv)
    root = repo_root()
    jobs = list(FAST_JOBS) if args.fast else (list(dict.fromkeys(args.job)) if args.job else list(ALL_JOBS))
    if args.list:
        print(CI_DOC, "jobs:", ", ".join(jobs))
        return 0
    env = _ci_env()
    try:

            for job in jobs:
                if job == "security":
                    py = _python()
                    _banner("Job: security")
                    _step("Install tools", [py, "-m", "pip", "install", "--upgrade", "detect-secrets", "pip-audit"], cwd=root, env=env)
                    _shell("Secret scan", 'set -euo pipefail\nif [ -f .secrets.baseline ]; then\n  git ls-files -z | xargs -0 detect-secrets-hook --baseline .secrets.baseline\nelse\n  detect-secrets scan . > .secrets.baseline\n  git diff --exit-code .secrets.baseline\nfi', cwd=root, env=env)
                    _shell("pip-audit", "if [ -f requirements.txt ]; then pip-audit -r requirements.txt --strict; else echo skip=no_requirements; fi", cwd=root, env=env)
                else:
                    raise SystemExit(f"unknown job: {job}")
    
    except subprocess.CalledProcessError as exc:
        print(f"Local quality gate failed ({exc.returncode})", file=sys.stderr)
        return exc.returncode
    print("\nOK: all selected jobs passed.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
