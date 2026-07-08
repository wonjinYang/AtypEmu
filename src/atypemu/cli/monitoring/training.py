"""Launch the Streamlit monitor for an AtypEmu training run."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import atypemu.monitoring.app as monitor_app


DEFAULT_REMOTE_BASE = Path("/scratch/wkyu514/yang07/atypemu")
DEFAULT_RUN_DIR = (
    DEFAULT_REMOTE_BASE / "results" / "baseline_ucbshift_rebuild_seed7_rerun"
)
DEFAULT_INTEGRATED_ROOT = DEFAULT_REMOTE_BASE / "data" / "integrated"


def main(argv: list[str] | None = None) -> None:
    """Run ``streamlit run`` for the training monitor app."""

    args = parse_args(argv)
    app_path = Path(monitor_app.__file__).resolve()
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--global.developmentMode=false",
        "--server.address",
        args.host,
        "--server.port",
        str(args.port),
        "--",
        "--run-dir",
        str(args.run_dir),
        "--integrated-root",
        str(args.integrated_root),
        "--refresh-seconds",
        str(args.refresh_seconds),
    ]
    if args.job_id:
        command.extend(["--job-id", str(args.job_id)])

    try:
        completed = subprocess.run(command, check=False)
    except FileNotFoundError as exc:
        raise SystemExit(
            "Streamlit is required for the monitor. Install with "
            "`pip install -e '.[monitor]'`."
        ) from exc
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse monitor launcher arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--integrated-root", default=str(DEFAULT_INTEGRATED_ROOT))
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8501)
    parser.add_argument("--refresh-seconds", type=int, default=30)
    return parser.parse_args(argv)


if __name__ == "__main__":
    main()
