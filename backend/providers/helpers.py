from __future__ import annotations

import io
import importlib.metadata
import os
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Iterable

from PIL import Image

from backend.providers.base import ProviderExecutionError


def velocity_root() -> Path:
    return Path.home() / ".velocity3d"


def base_dir() -> Path:
    configured = os.environ.get("VELOCITY_BASE_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path.cwd()


def model_root() -> Path:
    return base_dir() / "models"


def huggingface_root() -> Path:
    return base_dir() / "HuggingFace"


def huggingface_hub_cache() -> Path:
    return huggingface_root() / "hub"


def huggingface_transformers_cache() -> Path:
    return huggingface_root() / "transformers"


def checkpoints_root() -> Path:
    root = base_dir() / "Checkpoints"
    root.mkdir(parents=True, exist_ok=True)
    return root


def configure_huggingface_cache() -> Path:
    root = huggingface_root()
    hub_cache = huggingface_hub_cache()
    transformers_cache = huggingface_transformers_cache()
    diffusers_cache = root / "diffusers"
    datasets_cache = root / "datasets"
    assets_cache = root / "assets"
    torch_home = root / "torch"
    hy3dgen_models = root / "hy3dgen"

    for path in (root, hub_cache, transformers_cache, diffusers_cache, datasets_cache, assets_cache, torch_home, hy3dgen_models):
        path.mkdir(parents=True, exist_ok=True)

    os.environ["HF_HOME"] = str(root)
    os.environ["HF_HUB_CACHE"] = str(hub_cache)
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(transformers_cache)
    os.environ["DIFFUSERS_CACHE"] = str(diffusers_cache)
    os.environ["HF_DATASETS_CACHE"] = str(datasets_cache)
    os.environ["HF_ASSETS_CACHE"] = str(assets_cache)
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["TORCH_HOME"] = str(torch_home)
    os.environ["HY3DGEN_MODELS"] = str(hy3dgen_models)
    return root


def vendor_root() -> Path:
    return model_root()


def ensure_vendor_root() -> Path:
    root = vendor_root()
    root.mkdir(parents=True, exist_ok=True)
    return root


def repo_path(vendor_dir_name: str) -> Path:
    return vendor_root() / vendor_dir_name


def python_executable() -> str:
    home_venv = os.environ.get("HOME_VENV", "").strip()
    if home_venv:
        root = Path(home_venv)
        if sys.platform == "win32":
            return str(root / "Scripts" / "python.exe")
        return str(root / "bin" / "python3")
    return sys.executable


def current_python_series() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}"


def current_python_version() -> str:
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def installed_package_version(package_name: str) -> str | None:
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def safe_stem(value: str, fallback: str = "asset") -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in value.strip()[:48])
    return cleaned or fallback


def unique_output_path(output_dir: Path, stem: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir / f"{safe_stem(stem)}_{uuid.uuid4().hex[:8]}.glb"


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(image_bytes))
        return image.convert("RGBA")
    except Exception as exc:  # pragma: no cover - exercised through provider calls
        raise ProviderExecutionError(f"Could not decode image bytes: {exc}") from exc


def maybe_add_to_syspath(paths: Iterable[Path]) -> None:
    for path in paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


def run_subprocess(args: list[str], cwd: Path, timeout: int = 1800) -> subprocess.CompletedProcess[str]:
    try:
        return subprocess.run(
            args,
            cwd=str(cwd),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )
    except subprocess.TimeoutExpired as exc:
        raise ProviderExecutionError(f"Command timed out after {timeout}s: {' '.join(args)}") from exc
    except FileNotFoundError as exc:
        raise ProviderExecutionError(f"Command not found: {args[0]}") from exc


def export_scene_asset_to_glb(source_path: Path, output_path: Path) -> Path:
    import trimesh

    mesh = trimesh.load(str(source_path), force="scene")
    mesh.export(str(output_path))
    return output_path


def write_temp_image(image: Image.Image, suffix: str = ".png") -> Path:
    temp_dir = velocity_root() / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="velocity3d_", suffix=suffix, dir=str(temp_dir))
    os.close(fd)
    image.save(path)
    return Path(path)
