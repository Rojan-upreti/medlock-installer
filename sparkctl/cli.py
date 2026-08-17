"""Command-line entry: wizard, status, logs, stop, test, start."""

from __future__ import annotations

import argparse
import json
import sys

from sparkctl import CONTAINER_NAME, __version__
from sparkctl.serve import (
    container_logs,
    container_status,
    curl_example,
    load_config,
    public_endpoints,
    pull_image,
    smoke_test,
    start_container,
    stop_container,
    wait_healthy,
)


def _cmd_wizard(_args: argparse.Namespace) -> int:
    from sparkctl.tui import run_wizard

    run_wizard()
    return 0


def _cmd_status(_args: argparse.Namespace) -> int:
    status = container_status()
    cfg = load_config()
    print(json.dumps(status, indent=2))
    if cfg:
        print()
        print("Saved config:")
        print(f"  model:    {cfg.hf_repo or cfg.model_dir}")
        print(f"  served:   {cfg.served_name}")
        print(f"  endpoint: {cfg.endpoint()}")
        for url in public_endpoints(cfg)[1:]:
            print(f"            {url}")
    return 0 if status.get("running") == "yes" else 1


def _cmd_logs(args: argparse.Namespace) -> int:
    print(container_logs(tail=args.tail))
    return 0


def _cmd_stop(_args: argparse.Namespace) -> int:
    def log(msg: str) -> None:
        print(msg)

    stop_container(log)
    print(f"Stopped {CONTAINER_NAME}")
    return 0


def _cmd_test(_args: argparse.Namespace) -> int:
    cfg = load_config()
    if cfg is None:
        print("No saved config. Run: sparkctl wizard", file=sys.stderr)
        return 1
    ok, text = smoke_test(cfg)
    print(text)
    if ok:
        print()
        print(curl_example(cfg))
        return 0
    return 1


def _cmd_start(_args: argparse.Namespace) -> int:
    cfg = load_config()
    if cfg is None:
        print("No saved config. Run: sparkctl wizard", file=sys.stderr)
        return 1

    def log(msg: str) -> None:
        print(msg)

    try:
        pull_image(cfg, log)
        start_container(cfg, log)
        wait_healthy(cfg, log)
    except Exception as exc:  # noqa: BLE001
        print(exc, file=sys.stderr)
        return 1
    print()
    print(curl_example(cfg))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sparkctl",
        description="DGX Spark LLM installer and vLLM OpenAI-compatible server control.",
    )
    parser.add_argument("--version", action="version", version=f"sparkctl {__version__}")
    sub = parser.add_subparsers(dest="cmd")

    sub.add_parser("wizard", help="Interactive installer (default)")
    sub.add_parser("status", help="Show container and endpoint status")
    logs = sub.add_parser("logs", help="Show vLLM container logs")
    logs.add_argument("--tail", type=int, default=80, help="Number of log lines (default 80)")
    sub.add_parser("stop", help="Stop the spark-vllm container")
    sub.add_parser("start", help="Start vLLM from the last saved config")
    sub.add_parser("test", help="Send a sample chat completion request")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cmd = args.cmd or "wizard"
    handlers = {
        "wizard": _cmd_wizard,
        "status": _cmd_status,
        "logs": _cmd_logs,
        "stop": _cmd_stop,
        "start": _cmd_start,
        "test": _cmd_test,
    }
    return handlers[cmd](args)


if __name__ == "__main__":
    sys.exit(main())
