"""Pull the NGC vLLM image, run the server, and manage the container."""

from __future__ import annotations

import json
import os
import shutil
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx

from sparkctl import (
    CONFIG_DIR,
    CONFIG_FILE,
    CONTAINER_NAME,
    DEFAULT_GPU_MEMORY_UTILIZATION,
    DEFAULT_PORT,
    DEFAULT_SERVED_NAME,
    DEFAULT_VLLM_IMAGE,
    DEFAULT_VLLM_TAG,
    HF_CACHE_DIR,
    HEALTH_TIMEOUT_SEC,
)
from sparkctl.recipes import find_recipe
from sparkctl.util import docker_prefix, local_ips, run, stream_cmd, which


@dataclass
class ServeConfig:
    source: str = "huggingface"  # huggingface | local
    hf_repo: str | None = None
    model_dir: str | None = None
    served_name: str = DEFAULT_SERVED_NAME
    port: int = DEFAULT_PORT
    gpu_memory_utilization: float = DEFAULT_GPU_MEMORY_UTILIZATION
    max_model_len: int | None = None
    start_on_boot: bool = True
    trust_remote_code: bool = True
    extra_args: list[str] = field(default_factory=list)
    image: str = DEFAULT_VLLM_IMAGE
    tag: str = DEFAULT_VLLM_TAG
    hf_token_set: bool = False

    def __post_init__(self) -> None:
        self.image = os.environ.get("SPARKCTL_VLLM_IMAGE", self.image)
        self.tag = os.environ.get("SPARKCTL_VLLM_TAG", self.tag)

    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def model_arg(self) -> str:
        if self.source == "local" and self.model_dir:
            return "/models/current"
        if self.model_dir:
            return "/models/current"
        if self.hf_repo:
            return self.hf_repo
        raise ValueError("No model selected")


def load_config() -> ServeConfig | None:
    if not CONFIG_FILE.is_file():
        return None
    data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
    known = {k: v for k, v in data.items() if k in ServeConfig.__dataclass_fields__}
    return ServeConfig(**known)


def save_config(cfg: ServeConfig) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    payload = asdict(cfg)
    CONFIG_FILE.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def apply_recipe_defaults(cfg: ServeConfig) -> ServeConfig:
    if not cfg.hf_repo:
        return cfg
    recipe = find_recipe(cfg.hf_repo)
    if not recipe:
        return cfg
    if recipe.extra_args:
        # Keep user GPU util from the wizard; only merge non-conflicting flags.
        merged: list[str] = []
        args = list(recipe.extra_args)
        i = 0
        while i < len(args):
            if args[i] == "--gpu-memory-utilization":
                i += 2
                continue
            merged.append(args[i])
            i += 1
        existing = set(cfg.extra_args)
        for item in merged:
            if item not in existing:
                cfg.extra_args.append(item)
    cfg.trust_remote_code = recipe.trust_remote_code or cfg.trust_remote_code
    return cfg


def _docker() -> list[str]:
    return docker_prefix()


def image_ref(cfg: ServeConfig) -> str:
    return f"{cfg.image}:{cfg.tag}"


def ngc_login(api_key: str, on_line: Callable[[str], None] | None = None) -> None:
    key = api_key.strip()
    if not key:
        return
    log = on_line or (lambda _m: None)
    log("Logging into nvcr.io …")
    code, out = run(
        _docker() + ["login", "nvcr.io", "--username", "$oauthtoken", "--password-stdin"],
        timeout=60,
        input_text=key + "\n",
    )
    if code != 0:
        raise RuntimeError(f"NGC docker login failed: {out}")
    log("NGC login succeeded")


def pull_image(cfg: ServeConfig, on_line: Callable[[str], None]) -> None:
    ref = image_ref(cfg)
    on_line(f"Pulling {ref} (this can take several minutes) …")
    env = {"DOCKER_CLI_HINTS": "false"}
    code = stream_cmd(_docker() + ["pull", ref], on_line, env=env)
    if code != 0:
        raise RuntimeError(
            f"Failed to pull {ref}. "
            "If NGC requires auth, paste an NGC API key on the previous screen "
            "(https://ngc.nvidia.com) or run: docker login nvcr.io"
        )
    on_line(f"Image ready: {ref}")


def container_running() -> bool:
    try:
        code, out = run(
            _docker()
            + ["inspect", "-f", "{{.State.Running}}", CONTAINER_NAME],
            timeout=10,
        )
    except PermissionError:
        return False
    return code == 0 and out.strip().lower() == "true"


def container_exists() -> bool:
    code, _out = run(_docker() + ["inspect", CONTAINER_NAME], timeout=10)
    return code == 0


def stop_container(on_line: Callable[[str], None] | None = None) -> None:
    log = on_line or (lambda _m: None)
    if not container_exists():
        log("No existing spark-vllm container")
        return
    log("Stopping existing spark-vllm container …")
    run(_docker() + ["rm", "-f", CONTAINER_NAME], timeout=60)


def container_status() -> dict[str, str]:
    status = {
        "name": CONTAINER_NAME,
        "exists": "no",
        "running": "no",
        "status": "absent",
        "image": "",
        "ports": "",
    }
    if not which("docker"):
        status["status"] = "docker missing"
        return status
    try:
        prefix = _docker()
    except PermissionError as exc:
        status["status"] = str(exc)
        return status
    code, out = run(
        prefix
        + [
            "inspect",
            "-f",
            "{{.State.Status}}|{{.State.Running}}|{{.Config.Image}}",
            CONTAINER_NAME,
        ],
        timeout=10,
    )
    if code != 0:
        return status
    parts = out.split("|")
    status["exists"] = "yes"
    status["status"] = parts[0] if parts else out
    status["running"] = "yes" if len(parts) > 1 and parts[1].strip().lower() == "true" else "no"
    status["image"] = parts[2] if len(parts) > 2 else ""
    cfg = load_config()
    if cfg:
        status["ports"] = str(cfg.port)
        status["endpoint"] = cfg.endpoint()
        status["model"] = cfg.served_name
    return status


def container_logs(tail: int = 80) -> str:
    code, out = run(_docker() + ["logs", "--tail", str(tail), CONTAINER_NAME], timeout=20)
    return out if out else f"(no logs, exit {code})"


def build_docker_cmd(cfg: ServeConfig) -> list[str]:
    cmd = _docker() + [
        "run",
        "-d",
        "--name",
        CONTAINER_NAME,
        "--gpus",
        "all",
        "--ipc",
        "host",
        "--network",
        "host",
        "--ulimit",
        "memlock=-1",
        "--ulimit",
        "stack=67108864",
        "--restart",
        "unless-stopped" if cfg.start_on_boot else "no",
    ]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        cmd += ["-e", f"HF_TOKEN={token}"]
    HF_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cmd += ["-v", f"{HF_CACHE_DIR}:/root/.cache/huggingface"]
    if cfg.model_dir:
        cmd += ["-v", f"{cfg.model_dir}:/models/current:ro"]
    cmd += [image_ref(cfg), "vllm", "serve", cfg.model_arg()]
    cmd += [
        "--host",
        "0.0.0.0",
        "--port",
        str(cfg.port),
        "--served-model-name",
        cfg.served_name,
        "--gpu-memory-utilization",
        str(cfg.gpu_memory_utilization),
    ]
    if cfg.max_model_len:
        cmd += ["--max-model-len", str(cfg.max_model_len)]
    if cfg.trust_remote_code:
        cmd += ["--trust-remote-code"]
    cmd += list(cfg.extra_args)
    return cmd


def start_container(cfg: ServeConfig, on_line: Callable[[str], None]) -> None:
    stop_container(on_line)
    cmd = build_docker_cmd(cfg)
    printable = []
    skip = False
    for part in cmd:
        if skip:
            skip = False
            printable.append("HF_TOKEN=***")
            continue
        if part.startswith("HF_TOKEN="):
            printable.append("HF_TOKEN=***")
            continue
        if part == "-e":
            printable.append(part)
            skip = True
            continue
        printable.append(part)
    on_line("Starting vLLM: " + " ".join(printable))
    code, out = run(cmd, timeout=120)
    if code != 0:
        raise RuntimeError(f"docker run failed: {out}")
    on_line(f"Container {CONTAINER_NAME} started ({out.strip()[:12]})")


def wait_healthy(cfg: ServeConfig, on_line: Callable[[str], None], timeout: int = HEALTH_TIMEOUT_SEC) -> None:
    health = f"{cfg.endpoint()}/health"
    models = f"{cfg.endpoint()}/v1/models"
    on_line(f"Waiting for {health} (up to {timeout}s, model load can be slow) …")
    deadline = time.time() + timeout
    last_err = ""
    while time.time() < deadline:
        if not container_running():
            logs = container_logs(60)
            raise RuntimeError(f"Container exited while loading the model.\n{logs}")
        try:
            with httpx.Client(timeout=5.0) as client:
                health_resp = client.get(health)
                if health_resp.status_code == 200:
                    models_resp = client.get(models)
                    if models_resp.status_code == 200:
                        on_line("Endpoint is healthy")
                        return
                    last_err = f"/v1/models → {models_resp.status_code}"
                else:
                    last_err = f"/health → {health_resp.status_code}"
        except httpx.HTTPError as exc:
            last_err = str(exc)
        remaining = int(deadline - time.time())
        on_line(f"Still loading … {last_err or 'no response yet'} ({remaining}s left)")
        time.sleep(8)
    logs = container_logs(80)
    raise RuntimeError(f"Server did not become healthy in {timeout}s. Last error: {last_err}\n{logs}")


def smoke_test(cfg: ServeConfig) -> tuple[bool, str]:
    url = f"{cfg.endpoint()}/v1/chat/completions"
    payload = {
        "model": cfg.served_name,
        "messages": [{"role": "user", "content": "Reply with the single word: pong"}],
        "max_tokens": 16,
        "temperature": 0,
    }
    try:
        with httpx.Client(timeout=120.0) as client:
            resp = client.post(url, json=payload)
    except httpx.HTTPError as exc:
        return False, f"Request failed: {exc}"
    if resp.status_code != 200:
        return False, f"HTTP {resp.status_code}: {resp.text[:400]}"
    try:
        text = resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, ValueError):
        text = resp.text[:400]
    return True, text.strip()


def install_systemd(cfg: ServeConfig, on_line: Callable[[str], None]) -> None:
    if not cfg.start_on_boot:
        on_line("Start on boot disabled — skipping systemd")
        return
    if not which("systemctl"):
        on_line("systemctl not found; Docker --restart unless-stopped will still apply")
        return

    unit = f"""[Unit]
Description=DGX Spark vLLM OpenAI-compatible server
After=docker.service network-online.target
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart={shutil.which("docker") or "/usr/bin/docker"} start {CONTAINER_NAME}
ExecStop={shutil.which("docker") or "/usr/bin/docker"} stop {CONTAINER_NAME}
TimeoutStartSec=0

[Install]
WantedBy=multi-user.target
"""
    unit_path = Path("/tmp/spark-vllm.service")
    unit_path.write_text(unit, encoding="utf-8")

    dest = Path("/etc/systemd/system/spark-vllm.service")
    code, out = run(["sudo", "-n", "cp", str(unit_path), str(dest)], timeout=15)
    if code != 0:
        user_dir = Path.home() / ".config" / "systemd" / "user"
        user_dir.mkdir(parents=True, exist_ok=True)
        user_unit = user_dir / "spark-vllm.service"
        user_unit.write_text(unit.replace("WantedBy=multi-user.target", "WantedBy=default.target"), encoding="utf-8")
        run(["systemctl", "--user", "daemon-reload"], timeout=15)
        enable = run(["systemctl", "--user", "enable", "--now", "spark-vllm.service"], timeout=20)
        if enable[0] == 0:
            on_line(f"Enabled user systemd unit {user_unit}")
        else:
            on_line(
                "Could not install systemd (no passwordless sudo). "
                "Docker restart policy unless-stopped is already set."
            )
        return

    run(["sudo", "-n", "systemctl", "daemon-reload"], timeout=15)
    en = run(["sudo", "-n", "systemctl", "enable", "--now", "spark-vllm.service"], timeout=20)
    if en[0] == 0:
        on_line("Enabled systemd unit /etc/systemd/system/spark-vllm.service")
    else:
        on_line(f"Copied unit but enable failed: {en[1]}")


def public_endpoints(cfg: ServeConfig) -> list[str]:
    urls = [cfg.endpoint()]
    for ip in local_ips():
        urls.append(f"http://{ip}:{cfg.port}")
    return list(dict.fromkeys(urls))


def curl_example(cfg: ServeConfig) -> str:
    return (
        f"curl http://127.0.0.1:{cfg.port}/v1/chat/completions \\\n"
        f'  -H "Content-Type: application/json" \\\n'
        f"  -d '{{\n"
        f'    "model": "{cfg.served_name}",\n'
        f'    "messages": [{{"role": "user", "content": "Hello"}}]\n'
        f"  }}'"
    )
