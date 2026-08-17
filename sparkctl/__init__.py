"""DGX Spark LLM installer and vLLM OpenAI-compatible server control."""

from pathlib import Path

__version__ = "1.0.0"

APP_NAME = "sparkctl"
CONTAINER_NAME = "spark-vllm"
DEFAULT_VLLM_IMAGE = "nvcr.io/nvidia/vllm"
DEFAULT_VLLM_TAG = "26.05.post1-py3"
DEFAULT_PORT = 8000
DEFAULT_SERVED_NAME = "spark-llm"
DEFAULT_GPU_MEMORY_UTILIZATION = 0.80
HEALTH_TIMEOUT_SEC = 900

CONFIG_DIR = Path.home() / ".config" / "sparkctl"
CONFIG_FILE = CONFIG_DIR / "config.json"
MODELS_DIR = Path.home() / "models"
HF_CACHE_DIR = Path.home() / ".cache" / "huggingface"
