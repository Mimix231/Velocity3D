from __future__ import annotations

import os
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


class TrellisProvider(GenerationProvider):
    _image_pipeline = None
    _text_pipeline = None
    _load_lock = threading.Lock()

    def _repo_dir(self) -> Path:
        repo_dir = repo_path("trellis")
        if not repo_dir.exists():
            raise ProviderConfigurationError("TRELLIS is not downloaded yet.")
        return repo_dir

    def _get_image_pipeline(self):
        if self._image_pipeline is not None:
            return self._image_pipeline
        with self._load_lock:
            if self._image_pipeline is None:
                repo_dir = self._repo_dir()
                maybe_add_to_syspath((repo_dir,))
                os.environ.setdefault("SPCONV_ALGO", "native")
                try:
                    from trellis.pipelines import TrellisImageTo3DPipeline
                except ImportError as exc:  # pragma: no cover - depends on vendor stack
                    raise ProviderDependencyError(
                        "TRELLIS dependencies are missing. Follow the upstream setup steps."
                    ) from exc
                self._image_pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
                self._image_pipeline.cuda()
        return self._image_pipeline

    def _get_text_pipeline(self):
        if self._text_pipeline is not None:
            return self._text_pipeline
        with self._load_lock:
            if self._text_pipeline is None:
                repo_dir = self._repo_dir()
                maybe_add_to_syspath((repo_dir,))
                os.environ.setdefault("SPCONV_ALGO", "native")
                try:
                    from trellis.pipelines import TrellisTextTo3DPipeline
                except ImportError as exc:  # pragma: no cover - depends on vendor stack
                    raise ProviderDependencyError(
                        "TRELLIS text dependencies are missing. Follow the upstream setup steps."
                    ) from exc
                self._text_pipeline = TrellisTextTo3DPipeline.from_pretrained("microsoft/TRELLIS-text-xlarge")
                self._text_pipeline.cuda()
        return self._text_pipeline

    def _export_glb(self, outputs, output_dir: Path, stem: str) -> str:
        try:
            from trellis.utils import postprocessing_utils

            glb = postprocessing_utils.to_glb(
                outputs["gaussian"][0],
                outputs["mesh"][0],
                simplify=0.95,
                texture_size=1024,
            )
        except Exception as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderExecutionError(f"TRELLIS export failed: {exc}") from exc

        output_path = unique_output_path(output_dir, stem)
        glb.export(str(output_path))
        return str(output_path)

    def generate_text(
        self,
        prompt: str,
        output_dir: Path,
        cancellation_event: threading.Event,
    ) -> str:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before model execution")

        pipeline = self._get_text_pipeline()

        try:
            outputs = pipeline.run(prompt, seed=1)
        except Exception as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderExecutionError(f"TRELLIS text inference failed: {exc}") from exc

        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled after model execution")

        return self._export_glb(outputs, output_dir, f"trellis_{prompt}")

    def generate_image(
        self,
        image_bytes: bytes,
        output_dir: Path,
        cancellation_event: threading.Event,
        prompt: Optional[str] = None,
    ) -> str:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before model execution")

        image = decode_image_bytes(image_bytes).convert("RGB")
        pipeline = self._get_image_pipeline()

        try:
            outputs = pipeline.run(image, seed=1)
        except Exception as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderExecutionError(f"TRELLIS inference failed: {exc}") from exc

        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled after model execution")

        return self._export_glb(outputs, output_dir, f"trellis_{prompt or 'image'}")


class Trellis2Provider(GenerationProvider):
    _pipeline = None
    _load_lock = threading.Lock()

    def _repo_dir(self) -> Path:
        repo_dir = repo_path("trellis2")
        if not repo_dir.exists():
            raise ProviderConfigurationError("TRELLIS.2 is not downloaded yet.")
        return repo_dir

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        with self._load_lock:
            if self._pipeline is None:
                repo_dir = self._repo_dir()
                maybe_add_to_syspath((repo_dir, repo_dir / "o-voxel"))
                os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
                os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
                try:
                    from trellis2.pipelines import Trellis2ImageTo3DPipeline
                except ImportError as exc:  # pragma: no cover - depends on vendor stack
                    raise ProviderDependencyError(
                        "TRELLIS.2 dependencies are missing. Follow the upstream setup steps."
                    ) from exc
                self._pipeline = Trellis2ImageTo3DPipeline.from_pretrained("microsoft/TRELLIS.2-4B")
                self._pipeline.cuda()
        return self._pipeline

    def generate_image(
        self,
        image_bytes: bytes,
        output_dir: Path,
        cancellation_event: threading.Event,
        prompt: Optional[str] = None,
    ) -> str:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before model execution")

        image = decode_image_bytes(image_bytes).convert("RGB")
        pipeline = self._get_pipeline()

        try:
            import o_voxel

            mesh = pipeline.run(image)[0]
            mesh.simplify(16777216)
            glb = o_voxel.postprocess.to_glb(
                vertices=mesh.vertices,
                faces=mesh.faces,
                attr_volume=mesh.attrs,
                coords=mesh.coords,
                attr_layout=mesh.layout,
                voxel_size=mesh.voxel_size,
                aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
                decimation_target=1000000,
                texture_size=2048,
                remesh=True,
                remesh_band=1,
                remesh_project=0,
                verbose=False,
            )
        except Exception as exc:  # pragma: no cover - depends on vendor stack
            raise ProviderExecutionError(f"TRELLIS.2 inference failed: {exc}") from exc

        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled after model execution")

        output_path = unique_output_path(output_dir, f"trellis2_{prompt or 'image'}")
        glb.export(str(output_path), extension_webp=True)
        return str(output_path)
