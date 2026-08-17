"""Hardware and prerequisite probes for DGX Spark (GB10) + Docker/vLLM."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass, field

from sparkctl.util import run, which


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = False
    warn: bool = False


@dataclass
class HardwareReport:
    checks: list[Check] = field(default_factory=list)
    arch: str = ""
    gpu_name: str = ""
    is_linux: bool = False
    is_aarch64: bool = False
    is_gb10: bool = False
    docker_ok: bool = False

    @property
    def required_failed(self) -> list[Check]:
        return [c for c in self.checks if c.required and not c.ok]

    @property
    def ready(self) -> bool:
        return not self.required_failed


def _arch() -> Check:
    arch = platform.machine().lower()
    ok = arch in {"aarch64", "arm64"}
    return Check(
        name="CPU architecture",
        ok=ok,
        warn=not ok,
        required=False,
        detail=f"{arch}" + ("" if ok else " (DGX Spark is aarch64; continuing anyway)"),
    )


def _os() -> Check:
    system = platform.system()
    return Check(
        name="Operating system",
        ok=system == "Linux",
        required=True,
        detail=system if system == "Linux" else f"{system} — this installer must run on the Spark (Linux)",
    )


def _gpu() -> tuple[Check, str]:
    if not which("nvidia-smi"):
        return (
            Check(
                name="NVIDIA GPU",
                ok=False,
                required=True,
                detail="nvidia-smi not found. Install NVIDIA drivers / DGX OS stack.",
            ),
            "",
        )
    code, out = run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        timeout=15,
    )
    if code != 0 or not out:
        code2, out2 = run(["nvidia-smi"], timeout=15)
        detail = out2.splitlines()[0] if out2 else (out or "nvidia-smi failed")
        return (
            Check(name="NVIDIA GPU", ok=code2 == 0, required=True, detail=detail[:200]),
            "",
        )
    line = out.splitlines()[0].strip()
    name = line.split(",")[0].strip()
    is_spark = any(token in name.upper() for token in ("GB10", "SPARK"))
    detail = line
    if not is_spark:
        detail += " (not detected as GB10; Spark-tuned defaults still applied)"
    return (
        Check(
            name="NVIDIA GPU",
            ok=True,
            warn=not is_spark,
            required=True,
            detail=detail,
        ),
        name,
    )


def _cuda() -> Check:
    if which("nvcc"):
        code, out = run(["nvcc", "--version"], timeout=10)
        version_line = next((ln for ln in out.splitlines() if "release" in ln.lower()), out[-80:] if out else "")
        ok = code == 0
        warn = ok and "13." not in version_line
        detail = version_line.strip() or "nvcc present"
        if warn:
            detail += " (Spark expects CUDA 13.x)"
        return Check(
            name="CUDA toolkit",
            ok=ok,
            warn=warn,
            required=False,
            detail=detail,
        )
    code, out = run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        timeout=10,
    )
    if code == 0 and out:
        return Check(
            name="CUDA toolkit",
            ok=True,
            warn=True,
            required=False,
            detail=f"nvcc not found; NVIDIA driver {out.splitlines()[0].strip()} (container supplies CUDA)",
        )
    return Check(
        name="CUDA toolkit",
        ok=True,
        warn=True,
        required=False,
        detail="nvcc not found; the NGC vLLM container brings its own CUDA",
    )


def _docker() -> Check:
    if not which("docker"):
        return Check(
            name="Docker",
            ok=False,
            required=True,
            detail="docker not found. Install Docker Engine, then NVIDIA Container Toolkit.",
        )
    code, out = run(["docker", "ps"], timeout=15)
    if code != 0:
        hint = out[:300] if out else "docker ps failed"
        if "permission denied" in hint.lower():
            hint = "Permission denied. Run: sudo usermod -aG docker $USER && newgrp docker"
        return Check(name="Docker", ok=False, required=True, detail=hint)
    ver_code, ver = run(["docker", "--version"], timeout=10)
    return Check(
        name="Docker",
        ok=True,
        required=True,
        detail=ver if ver_code == 0 else "docker ps succeeded",
    )


def _nvidia_runtime() -> Check:
    if not which("docker"):
        return Check(
            name="NVIDIA Container Toolkit",
            ok=False,
            required=True,
            detail="Cannot check: Docker is missing.",
        )
    code, out = run(["docker", "info"], timeout=20)
    if code == 0 and "nvidia" in out.lower():
        return Check(
            name="NVIDIA Container Toolkit",
            ok=True,
            required=True,
            detail="nvidia runtime registered with Docker",
        )
    if which("nvidia-container-runtime") or which("nvidia-container-cli"):
        return Check(
            name="NVIDIA Container Toolkit",
            ok=True,
            warn=True,
            required=True,
            detail="toolkit binaries found; Docker info did not list nvidia (GPU run may still work)",
        )
    return Check(
        name="NVIDIA Container Toolkit",
        ok=False,
        required=True,
        detail="nvidia runtime not visible. Install nvidia-container-toolkit and restart Docker.",
    )


def _python() -> Check:
    ver = platform.python_version()
    major, minor = platform.python_version_tuple()[:2]
    ok = int(major) >= 3 and int(minor) >= 10
    return Check(
        name="Python",
        ok=ok,
        required=True,
        detail=f"{ver}" + ("" if ok else " (need 3.10+)"),
    )


def _disk() -> Check:
    code, out = run(["df", "-h", os.path.expanduser("~")], timeout=5)
    detail = out.splitlines()[-1] if out else "unknown"
    return Check(name="Home disk space", ok=True, required=False, detail=detail)


def probe() -> HardwareReport:
    report = HardwareReport(
        arch=platform.machine(),
        is_linux=platform.system() == "Linux",
        is_aarch64=platform.machine().lower() in {"aarch64", "arm64"},
    )
    os_check = _os()
    arch_check = _arch()
    gpu_check, gpu_name = _gpu()
    docker_check = _docker()
    report.gpu_name = gpu_name
    report.is_gb10 = any(token in gpu_name.upper() for token in ("GB10", "SPARK"))
    report.docker_ok = docker_check.ok
    report.checks = [
        os_check,
        arch_check,
        gpu_check,
        _cuda(),
        docker_check,
        _nvidia_runtime(),
        _python(),
        _disk(),
    ]
    return report
