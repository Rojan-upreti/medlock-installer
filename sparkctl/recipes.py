"""Spark-validated model catalog and vLLM flag recipes."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Recipe:
    name: str
    repo_id: str
    quantization: str
    extra_args: list[str] = field(default_factory=list)
    trust_remote_code: bool = True
    notes: str = ""
    recommended: bool = False


# Curated from NVIDIA DGX Spark vLLM playbook model support matrix.
CATALOG: tuple[Recipe, ...] = (
    Recipe(
        name="Qwen2.5 Math 1.5B (first-run test)",
        repo_id="Qwen/Qwen2.5-Math-1.5B-Instruct",
        quantization="BF16",
        notes="Small playbook smoke-test model. Fast to download and a good first endpoint.",
        recommended=True,
    ),
    Recipe(
        name="GPT-OSS 20B",
        repo_id="openai/gpt-oss-20b",
        quantization="MXFP4",
        notes="OpenAI open-weight 20B. Fits comfortably in Spark unified memory.",
    ),
    Recipe(
        name="GPT-OSS 120B",
        repo_id="openai/gpt-oss-120b",
        quantization="MXFP4",
        extra_args=["--gpu-memory-utilization", "0.90"],
        notes="Large MXFP4 model. Uses most of the 128GB unified memory.",
    ),
    Recipe(
        name="Qwen3 8B NVFP4",
        repo_id="nvidia/Qwen3-8B-NVFP4",
        quantization="NVFP4",
        notes="NVIDIA NVFP4 checkpoint — best throughput on GB10.",
    ),
    Recipe(
        name="Qwen3 8B FP8",
        repo_id="nvidia/Qwen3-8B-FP8",
        quantization="FP8",
    ),
    Recipe(
        name="Qwen3 14B NVFP4",
        repo_id="nvidia/Qwen3-14B-NVFP4",
        quantization="NVFP4",
    ),
    Recipe(
        name="Qwen3 32B NVFP4",
        repo_id="nvidia/Qwen3-32B-NVFP4",
        quantization="NVFP4",
        extra_args=["--gpu-memory-utilization", "0.85"],
    ),
    Recipe(
        name="Llama 3.1 8B Instruct NVFP4",
        repo_id="nvidia/Llama-3.1-8B-Instruct-NVFP4",
        quantization="NVFP4",
    ),
    Recipe(
        name="Llama 3.1 8B Instruct FP8",
        repo_id="nvidia/Llama-3.1-8B-Instruct-FP8",
        quantization="FP8",
    ),
    Recipe(
        name="Llama 3.3 70B Instruct NVFP4",
        repo_id="nvidia/Llama-3.3-70B-Instruct-NVFP4",
        quantization="NVFP4",
        extra_args=["--gpu-memory-utilization", "0.90"],
        notes="70B NVFP4. Gated on Hugging Face — provide a token.",
    ),
    Recipe(
        name="Nemotron 3 Nano 30B FP8",
        repo_id="nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8",
        quantization="FP8",
        extra_args=["--gpu-memory-utilization", "0.85"],
    ),
    Recipe(
        name="Nemotron 3 Super 120B NVFP4",
        repo_id="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4",
        quantization="NVFP4",
        extra_args=[
            "--gpu-memory-utilization",
            "0.85",
            "--max-num-seqs",
            "4",
        ],
        notes="Very large MoE. Leave max context moderate if memory is tight.",
    ),
    Recipe(
        name="Gemma 4 E4B IT",
        repo_id="google/gemma-4-E4B-it",
        quantization="BF16",
        notes="Gemma 4 may need the gemma4 NGC/vLLM image on some tags.",
    ),
    Recipe(
        name="Phi-4 reasoning plus NVFP4",
        repo_id="nvidia/Phi-4-reasoning-plus-NVFP4",
        quantization="NVFP4",
        trust_remote_code=True,
    ),
)


def find_recipe(repo_id: str) -> Recipe | None:
    wanted = repo_id.strip().lower()
    for recipe in CATALOG:
        if recipe.repo_id.lower() == wanted:
            return recipe
    return None


def default_recipe() -> Recipe:
    for recipe in CATALOG:
        if recipe.recommended:
            return recipe
    return CATALOG[0]
