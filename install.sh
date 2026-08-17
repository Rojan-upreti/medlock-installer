#!/usr/bin/env bash
# Bootstrap the DGX Spark LLM installer, then launch the TUI wizard.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
VENV="${ROOT}/.venv"
BOOTSTRAP_ONLY=0

usage() {
  cat <<'EOF'
Usage: ./install.sh [--bootstrap-only]

Install Python deps, wire up the sparkctl command, and launch the
interactive wizard that downloads an LLM and starts a local vLLM
OpenAI-compatible endpoint on NVIDIA DGX Spark (GB10).

  --bootstrap-only   Create the venv and sparkctl wrapper, do not open the TUI
  -h, --help         Show this help

After install:
  sparkctl wizard | status | logs | test | stop | start
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --bootstrap-only) BOOTSTRAP_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *)
      echo "Unknown option: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

log() { printf '==> %s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }

need_python() {
  local cmd ver
  for cmd in python3.12 python3.11 python3; do
    if command -v "${cmd}" >/dev/null 2>&1; then
      ver="$("${cmd}" -c 'import sys; print("%d.%d" % (sys.version_info.major, sys.version_info.minor))')"
      if "${cmd}" -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
        echo "${cmd}"
        return 0
      fi
      warn "${cmd} is ${ver}; need Python 3.10+"
    fi
  done
  return 1
}

OS="$(uname -s)"
ARCH="$(uname -m)"
if [[ "${OS}" != "Linux" ]]; then
  warn "This installer is meant to run on the DGX Spark (Linux/${ARCH}). Continuing with a local venv only."
fi

log "DGX Spark LLM installer  (${OS} ${ARCH})"

if ! PY="$(need_python)"; then
  if [[ "${OS}" == "Linux" ]] && command -v apt-get >/dev/null 2>&1; then
    log "Installing Python via apt (requires sudo)"
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-venv python3-pip curl ca-certificates
    PY="$(need_python)" || {
      echo "error: Python 3.10+ is required" >&2
      exit 1
    }
  else
    echo "error: Python 3.10+ is required" >&2
    exit 1
  fi
fi
log "Using ${PY} ($("${PY}" -c 'import sys; print(sys.version.split()[0])'))"

if ! "${PY}" -c "import venv" >/dev/null 2>&1; then
  if [[ "${OS}" == "Linux" ]] && command -v apt-get >/dev/null 2>&1; then
    log "Installing python3-venv (requires sudo)"
    sudo apt-get install -y python3-venv python3-pip
  else
    echo "error: Python venv module is missing" >&2
    exit 1
  fi
fi

if [[ ! -x "${VENV}/bin/python" ]]; then
  log "Creating virtualenv at ${VENV}"
  "${PY}" -m venv "${VENV}"
fi

log "Installing Python packages"
"${VENV}/bin/python" -m pip install --upgrade pip
"${VENV}/bin/python" -m pip install -r "${ROOT}/requirements.txt"

# huggingface_hub ships the `hf` CLI; expose it from the venv.
if [[ -x "${VENV}/bin/hf" ]]; then
  log "Hugging Face CLI: ${VENV}/bin/hf"
fi

if ! command -v docker >/dev/null 2>&1; then
  warn "Docker is not on PATH. The wizard needs Docker + NVIDIA Container Toolkit."
  warn "On DGX OS / Ubuntu: install Docker Engine, then nvidia-container-toolkit."
elif ! docker ps >/dev/null 2>&1; then
  warn "Cannot talk to the Docker daemon. If you see permission denied:"
  warn "  sudo usermod -aG docker \$USER"
  warn "  newgrp docker     # or log out and back in"
else
  log "Docker is reachable"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null | while read -r line; do
    log "GPU: ${line}"
  done || true
else
  warn "nvidia-smi not found. Run this on the Spark (or a Linux box with NVIDIA drivers)."
fi

BIN_DIR="${HOME}/.local/bin"
mkdir -p "${BIN_DIR}"
cat > "${BIN_DIR}/sparkctl" <<EOF
#!/usr/bin/env bash
set -euo pipefail
export PYTHONPATH="${ROOT}\${PYTHONPATH:+:\$PYTHONPATH}"
exec "${VENV}/bin/python" -m sparkctl "\$@"
EOF
chmod +x "${BIN_DIR}/sparkctl" "${ROOT}/sparkctl.sh" "${ROOT}/install.sh"

log "Installed sparkctl → ${BIN_DIR}/sparkctl"
if ! command -v sparkctl >/dev/null 2>&1; then
  warn "${BIN_DIR} is not on PATH. Add this to your shell rc:"
  warn "  export PATH=\"\$HOME/.local/bin:\$PATH\""
fi

if [[ "${BOOTSTRAP_ONLY}" -eq 1 ]]; then
  log "Bootstrap complete. Run: ${BIN_DIR}/sparkctl wizard"
  exit 0
fi

if [[ -z "${TERM:-}" || "${TERM}" == "dumb" ]]; then
  warn "TERM is '${TERM:-unset}'. The TUI needs a real terminal (SSH with a TTY is fine)."
fi

log "Launching installer wizard"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec "${VENV}/bin/python" -m sparkctl wizard
