"""Hugging Face search, download, auth, and local checkpoint validation."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from sparkctl import MODELS_DIR

_DEVNULL = open(os.devnull, "w")


WEIGHT_GLOBS = ("*.safetensors", "*.bin", "*.pt", "*.pth", "*.npz", "*.gguf")


@dataclass
class ModelHit:
    repo_id: str
    downloads: int | None
    likes: int | None
    pipeline: str
    private: bool


@dataclass
class LocalModel:
    path: Path
    ok: bool
    detail: str
    weight_files: int = 0


def slugify_repo(repo_id: str) -> str:
    return repo_id.strip().replace("/", "--")


def model_dir_for(repo_id: str) -> Path:
    return MODELS_DIR / slugify_repo(repo_id)


def load_hf_token() -> str | None:
    env = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if env:
        return env.strip()
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    if token_path.is_file():
        value = token_path.read_text(encoding="utf-8").strip()
        return value or None
    return None


def save_hf_token(token: str) -> None:
    value = token.strip()
    if not value:
        return
    token_path = Path.home() / ".cache" / "huggingface" / "token"
    token_path.parent.mkdir(parents=True, exist_ok=True)
    token_path.write_text(value + "\n", encoding="utf-8")
    os.environ["HF_TOKEN"] = value
    os.environ["HUGGING_FACE_HUB_TOKEN"] = value


def search_models(query: str, token: str | None = None, limit: int = 20) -> list[ModelHit]:
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    hits: list[ModelHit] = []
    try:
        models = api.list_models(
            search=query.strip(),
            limit=limit,
            sort="downloads",
        )
    except TypeError:
        models = api.list_models(search=query.strip(), limit=limit)
    for info in models:
        hits.append(
            ModelHit(
                repo_id=info.id,
                downloads=getattr(info, "downloads", None),
                likes=getattr(info, "likes", None),
                pipeline=getattr(info, "pipeline_tag", None) or "",
                private=bool(getattr(info, "private", False)),
            )
        )
    return hits


def validate_local_model(path: str | Path) -> LocalModel:
    folder = Path(path).expanduser().resolve()
    if not folder.exists():
        return LocalModel(path=folder, ok=False, detail="Path does not exist")
    if folder.is_file():
        folder = folder.parent
    if not folder.is_dir():
        return LocalModel(path=folder, ok=False, detail="Not a directory")

    has_config = (folder / "config.json").is_file()
    weights: list[Path] = []
    for pattern in WEIGHT_GLOBS:
        weights.extend(folder.glob(pattern))
        weights.extend(folder.glob(f"*/{pattern}"))
    unique = {p.resolve() for p in weights}

    if not has_config and not unique:
        return LocalModel(
            path=folder,
            ok=False,
            detail="Need a Hugging Face-format folder (config.json + weight files)",
        )
    if not has_config:
        return LocalModel(
            path=folder,
            ok=False,
            detail="Missing config.json — vLLM expects a Transformers/HF checkpoint",
            weight_files=len(unique),
        )
    if not unique:
        return LocalModel(
            path=folder,
            ok=False,
            detail="config.json found, but no weight files (.safetensors / .bin)",
        )
    return LocalModel(
        path=folder,
        ok=True,
        detail=f"Valid checkpoint ({len(unique)} weight files)",
        weight_files=len(unique),
    )


def download_model(
    repo_id: str,
    dest: Path,
    token: str | None = None,
    on_progress: Callable[[str], None] | None = None,
) -> Path:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm as tqdm_auto

    dest.mkdir(parents=True, exist_ok=True)
    if on_progress:
        on_progress(f"Downloading {repo_id} → {dest}")

    class ProgressTqdm(tqdm_auto):
        def __init__(self, *args, **kwargs):
            kwargs["file"] = _DEVNULL
            kwargs.setdefault("mininterval", 0.5)
            super().__init__(*args, **kwargs)

        def update(self, n: float | int = 1):
            result = super().update(n)
            if on_progress:
                desc = str(getattr(self, "desc", None) or repo_id)
                total = self.total or 0
                if total:
                    pct = min(100, int(self.n * 100 / total))
                    on_progress(f"{desc} {pct}% ({self.n}/{total})")
                else:
                    on_progress(f"{desc} {self.n}")
            return result

    snapshot_download(
        repo_id=repo_id.strip(),
        token=token,
        local_dir=str(dest),
        max_workers=8,
        tqdm_class=ProgressTqdm,
    )
    if on_progress:
        on_progress(f"Finished downloading {repo_id}")
    return dest
