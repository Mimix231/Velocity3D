from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Optional

from backend.providers.base import (
    GenerationProvider,
    ProviderConfigurationError,
    ProviderDependencyError,
    ProviderExecutionError,
)
from backend.providers.helpers import (
    decode_image_bytes,
    maybe_add_to_syspath,
    repo_path,
    unique_output_path,
)


class _HunyuanBaseProvider(GenerationProvider):
    _pipeline = None
    _load_lock = threading.Lock()
    repo_name = ""

    def _repo_dir(self) -> Path:
        repo_dir = repo_path(self.repo_name)
        if not repo_dir.exists():
            raise ProviderConfigurationError(
                f"{self.model_id} is not downloaded yet. Open Model Library and clone the upstream repo first."
            )
        return repo_dir

    def _load_pipeline(self):
        raise NotImplementedError

    def _check_runtime_imports(self, module_names: tuple[str, ...]) -> None:
        missing: list[str] = []
        for module_name in module_names:
            try:
                importlib.import_module(module_name)
            except ImportError as exc:
                missing.append(f"{module_name} ({exc})")

        if missing:
            raise ProviderDependencyError(
                f"{self.model_id} is missing runtime dependencies: {', '.join(missing)}. "
                "Run Install / Download for this model, then restart the backend if pip replaced loaded packages."
            )

    def _get_pipeline(self):
        cls = type(self)
        if cls._pipeline is not None:
            return cls._pipeline
        with cls._load_lock:
            if cls._pipeline is None:
                cls._pipeline = self._load_pipeline()
        return cls._pipeline

    def release_pipeline(self) -> None:
        cls = type(self)
        with cls._load_lock:
            cls._pipeline = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def generate_image(
        self,
        image_bytes: bytes,
        output_dir: Path,
        cancellation_event: threading.Event,
        prompt: Optional[str] = None,
    ) -> str:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before model execution")

        image = decode_image_bytes(image_bytes)
        pipeline = self._get_pipeline()

        try:
            mesh = pipeline(image=image)[0]
        except Exception as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderExecutionError(f"{self.model_id} inference failed: {exc}") from exc

        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled after model execution")

        output_path = unique_output_path(output_dir, f"{self.model_id}_{prompt or 'image'}")
        try:
            mesh.export(str(output_path))
        except Exception as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderExecutionError(f"{self.model_id} export failed: {exc}") from exc
        return str(output_path)


class Hunyuan21Provider(_HunyuanBaseProvider):
    repo_name = "hunyuan3d-2.1"

    def _load_pipeline(self):
        repo_dir = self._repo_dir()
        maybe_add_to_syspath((repo_dir / "hy3dshape", repo_dir))
        self._check_runtime_imports(
            (
                "torch",
                "torchvision",
                "diffusers",
                "transformers",
                "yaml",
                "PIL",
                "einops",
                "cv2",
                "trimesh",
                "skimage",
                "timm",
                "torchdiffeq",
            )
        )
        try:
            from hy3dshape.pipelines import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderDependencyError(
                f"Hunyuan3D-2.1 dependencies are missing: {exc}. Run Install / Download in Model Library."
            ) from exc

        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2.1")


class Hunyuan20Provider(_HunyuanBaseProvider):
    repo_name = "hunyuan3d-2"

    def _load_pipeline(self):
        repo_dir = self._repo_dir()
        maybe_add_to_syspath((repo_dir,))
        self._check_runtime_imports(
            (
                "torch",
                "torchvision",
                "diffusers",
                "transformers",
                "yaml",
                "PIL",
                "einops",
                "cv2",
                "trimesh",
                "skimage",
                "timm",
                "torchdiffeq",
            )
        )
        try:
            from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline
        except ImportError as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderDependencyError(
                f"Hunyuan3D-2 dependencies are missing: {exc}. Run Install / Download in Model Library."
            ) from exc

        return Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2")
