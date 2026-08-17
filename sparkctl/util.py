"""Small process helpers shared by hardware probes and Docker/vLLM control."""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Callable
from pathlib import Path


def which(name: str) -> str | None:
    return shutil.which(name)


def run(
    cmd: list[str],
    *,
    timeout: int | None = 30,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
    input_text: str | None = None,
) -> tuple[int, str]:
    try:
        completed = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
            cwd=cwd,
            input=input_text,
            check=False,
        )
    except FileNotFoundError:
        return 127, f"command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, f"timed out: {' '.join(cmd)}"
    out = (completed.stdout or "") + (completed.stderr or "")
    return completed.returncode, out.strip()


def stream_cmd(
    cmd: list[str],
    on_line: Callable[[str], None],
    *,
    env: dict[str, str] | None = None,
    cwd: str | Path | None = None,
) -> int:
    """Run a command and stream combined output, turning CR progress into lines."""
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=merged_env,
        )
    except FileNotFoundError:
        on_line(f"command not found: {cmd[0]}")
        return 127

    assert proc.stdout is not None
    buf = b""
    while True:
        chunk = proc.stdout.read(256)
        if not chunk:
            break
        buf += chunk.replace(b"\r", b"\n")
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            text = line.decode("utf-8", "replace").strip()
            if text:
                on_line(text)
    if buf.strip():
        on_line(buf.decode("utf-8", "replace").strip())
    return proc.wait()


def docker_prefix() -> list[str]:
    """Return a docker argv prefix. Never uses sudo (TUI cannot prompt)."""
    if not which("docker"):
        return ["docker"]
    code, out = run(["docker", "ps"], timeout=10)
    if code == 0:
        return ["docker"]
    if "permission denied" in out.lower():
        raise PermissionError(
            "Docker is installed but this user cannot talk to the daemon. "
            "Run: sudo usermod -aG docker $USER   then log out and back in."
        )
    return ["docker"]


def local_ips() -> list[str]:
    code, out = run(["hostname", "-I"], timeout=5)
    if code == 0 and out:
        return [ip for ip in out.split() if ip and ip != "127.0.0.1"]
    return []
