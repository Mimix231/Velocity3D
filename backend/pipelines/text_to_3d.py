"""
TextTo3DPipeline: generates a 3D mesh from a text prompt using Shap-E.
Requires: pip install git+https://github.com/openai/shap-e.git
"""
from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np

from backend.pipelines.bpy_post import BpyPostProcessor, BpyPostProcessorError
from backend.providers.helpers import huggingface_root

logger = logging.getLogger("velocity3d.text_to_3d")

CACHE_DIR = huggingface_root() / "shap_e"


class TextTo3DError(Exception):
    pass


class TextTo3DPipeline:
    """
    Generates a 3D mesh from a text prompt using Shap-E.
    Weights are downloaded on first use and cached in ~/.velocity3d/cache/shap_e/.
    """

    _model = None
    _diffusion = None
    _xm = None
    _load_lock = threading.Lock()

    def _load_models(self) -> None:
        """Load Shap-E models (lazy, cached)."""
        if self._model is not None:
            return

        with self._load_lock:
            if self._model is not None:
                return

            logger.info("Loading Shap-E models (first run may download weights)...")
            try:
                import torch
                from shap_e.diffusion.sample import sample_latents
                from shap_e.diffusion.gaussian_diffusion import diffusion_from_config
                from shap_e.models.download import load_model, load_config
                from shap_e.util.notebooks import decode_latent_mesh

                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                CACHE_DIR.mkdir(parents=True, exist_ok=True)

                xm = load_model("transmitter", device=device, cache_dir=str(CACHE_DIR))
                model = load_model("text300M", device=device, cache_dir=str(CACHE_DIR))
                diffusion = diffusion_from_config(load_config("diffusion", cache_dir=str(CACHE_DIR)))

                TextTo3DPipeline._xm = xm
                TextTo3DPipeline._model = model
                TextTo3DPipeline._diffusion = diffusion
                TextTo3DPipeline._device = device

                logger.info("Shap-E models loaded on %s", device)
            except ImportError as exc:
                raise TextTo3DError(
                    "shap-e is not installed. Install with: "
                    "pip install git+https://github.com/openai/shap-e.git"
                ) from exc

    def generate(
        self,
        prompt: str,
        output_dir: Path,
        cancellation_event: threading.Event,
        guidance_scale: float = 15.0,
        num_steps: int = 64,
    ) -> str:
        """
        Generate a 3D mesh from a text prompt.

        Args:
            prompt: natural language description of the 3D model
            output_dir: directory to write the output GLB
            cancellation_event: set this to cancel generation
            guidance_scale: classifier-free guidance scale
            num_steps: number of diffusion steps

        Returns:
            Absolute path to the generated .glb file
        """
        start = time.time()
        logger.info("TextTo3DPipeline.generate: prompt=%r", prompt)

        self._load_models()

        if cancellation_event.is_set():
            raise TextTo3DError("Generation cancelled before inference")

        try:
            import torch
            from shap_e.diffusion.sample import sample_latents
            from shap_e.util.notebooks import decode_latent_mesh

            device = TextTo3DPipeline._device

            batch_size = 1
            latents = sample_latents(
                batch_size=batch_size,
                model=TextTo3DPipeline._model,
                diffusion=TextTo3DPipeline._diffusion,
                guidance_scale=guidance_scale,
                model_kwargs=dict(texts=[prompt] * batch_size),
                progress=True,
                clip_denoised=True,
                use_fp16=True,
                use_karras=True,
                karras_steps=num_steps,
                sigma_min=1e-3,
                sigma_max=160,
                s_churn=0,
            )

            if cancellation_event.is_set():
                raise TextTo3DError("Generation cancelled after inference")

            # Decode latent to mesh
            t = decode_latent_mesh(TextTo3DPipeline._xm, latents[0]).tri_mesh()
            vertices = np.array(t.verts, dtype=np.float32)
            faces = np.array(t.faces, dtype=np.int32)

        except TextTo3DError:
            raise
        except Exception as exc:
            raise TextTo3DError(f"Shap-E inference failed: {exc}") from exc

        # Post-process with bpy
        processor = BpyPostProcessor()
        safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
        output_path = processor.process(vertices, faces, output_dir, name=safe_name or "mesh")

        elapsed = time.time() - start
        logger.info("TextTo3DPipeline.generate: done in %.1fs -> %s", elapsed, output_path)
        return output_path
