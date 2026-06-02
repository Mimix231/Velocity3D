from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Optional

from backend.providers.base import (
    GenerationProvider,
    ProviderConfigurationError,
    ProviderExecutionError,
)
from backend.providers.helpers import (
    decode_image_bytes,
    export_scene_asset_to_glb,
    python_executable,
    repo_path,
    run_subprocess,
    unique_output_path,
    write_temp_image,
)


class _RepoRunScriptProvider(GenerationProvider):
    repo_name = ""
    repo_script = "run.py"

    def _repo_dir(self) -> Path:
        repo_dir = repo_path(self.repo_name)
        if not repo_dir.exists():
            raise ProviderConfigurationError(
                f"{self.model_id} is not downloaded yet. Clone the upstream repo from Model Library first."
            )
        return repo_dir

    def _build_args(self, image_path: Path, output_dir: Path) -> list[str]:
        return [python_executable(), self.repo_script, str(image_path), "--output-dir", str(output_dir)]

    def generate_image(
        self,
        image_bytes: bytes,
        output_dir: Path,
        cancellation_event: threading.Event,
        prompt: Optional[str] = None,
        pipeline_options: Any = None,
    ) -> str:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before model execution")

        image = decode_image_bytes(image_bytes).convert("RGB")
        temp_input = write_temp_image(image)
        run_output_dir = output_dir / f"{self.model_id}_{temp_input.stem}"
        run_output_dir.mkdir(parents=True, exist_ok=True)

        result = run_subprocess(self._build_args(temp_input, run_output_dir), cwd=self._repo_dir())
        if result.returncode != 0:
            raise ProviderExecutionError(
                f"{self.model_id} CLI failed with code {result.returncode}\n"
                f"{(result.stderr or result.stdout).strip()[:2000]}"
            )

        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled after model execution")

        asset = self._find_generated_asset(run_output_dir)
        if asset is None:
            raise ProviderExecutionError(f"{self.model_id} produced no exportable asset in {run_output_dir}")

        if asset.suffix.lower() == ".glb":
            return str(asset)

        output_path = unique_output_path(output_dir, f"{self.model_id}_{prompt or 'image'}")
        export_scene_asset_to_glb(asset, output_path)
        return str(output_path)

    def _find_generated_asset(self, output_dir: Path) -> Path | None:
        preferred_exts = (".glb", ".obj", ".ply")
        for ext in preferred_exts:
            matches = sorted(output_dir.glob(f"*{ext}"))
            if matches:
                return matches[0]
        return None


class StableFast3DProvider(_RepoRunScriptProvider):
    repo_name = "stable-fast-3d"


class TripoSRProvider(_RepoRunScriptProvider):
    repo_name = "triposr"
