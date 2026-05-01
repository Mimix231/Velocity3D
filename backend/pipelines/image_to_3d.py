"""
ImageTo3DPipeline — Python 3.13 compatible image-to-3D generation.

Stack (all pip-installable, all Python 3.13 compatible):
  - transformers  : Depth Anything V2 for monocular depth estimation
  - scikit-image  : marching_cubes for depth-volume → triangle mesh
  - scipy         : gaussian smoothing of depth volume
  - trimesh       : mesh cleanup and GLB export fallback
  - numpy, Pillow : image/array handling

No open3d, no TripoSR, no cloning required.
"""
from __future__ import annotations

import io
import logging
import threading
import time
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image

from backend.pipelines.bpy_post import BpyPostProcessor, BpyPostProcessorError
from backend.providers.helpers import huggingface_transformers_cache

logger = logging.getLogger("velocity3d.image_to_3d")

CACHE_DIR = huggingface_transformers_cache() / "depth_anything"

# Depth Anything V2 Small — ~100 MB, fast, Python 3.13 compatible via transformers
DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


class UnsupportedImageError(Exception):
    pass


class ImageTo3DError(Exception):
    pass


class ImageTo3DPipeline:
    """
    Generates a 3D mesh from a single image.

    Pipeline:
      1. Depth Anything V2 (transformers) → depth map
      2. Build a 3D occupancy volume from the depth map
      3. scikit-image marching_cubes → triangle mesh
      4. BpyPostProcessor (subprocess) → cleaned GLB
    """

    _pipe = None
    _load_lock = threading.Lock()

    def _load_model(self) -> None:
        if self._pipe is not None:
            return
        with self._load_lock:
            if self._pipe is not None:
                return
            logger.info("Loading Depth Anything V2 (~100 MB on first run)...")
            try:
                from transformers import pipeline as hf_pipeline
                CACHE_DIR.mkdir(parents=True, exist_ok=True)
                pipe = hf_pipeline(
                    task="depth-estimation",
                    model=DEPTH_MODEL_ID,
                    cache_dir=str(CACHE_DIR),
                )
                ImageTo3DPipeline._pipe = pipe
                logger.info("Depth Anything V2 ready.")
            except ImportError as exc:
                raise ImageTo3DError(
                    "transformers is not installed. Run: pip install transformers"
                ) from exc

    def generate(
        self,
        image_bytes: bytes,
        output_dir: Path,
        cancellation_event: threading.Event,
        prompt: Optional[str] = None,
    ) -> str:
        """
        Generate a 3D mesh from an image.

        Args:
            image_bytes : raw image bytes (JPEG, PNG, WEBP, BMP)
            output_dir  : directory to write the output GLB
            cancellation_event : set to cancel
            prompt      : optional text hint (reserved for future refinement)

        Returns:
            Absolute path to the generated .glb file
        """
        start = time.time()
        logger.info("ImageTo3DPipeline.generate: prompt=%r", prompt)

        # ── Decode image ──────────────────────────────────────────────────────
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            raise UnsupportedImageError(f"Cannot decode image: {exc}") from exc

        # Resize to 512px on the long edge — good balance of quality vs speed
        max_dim = 512
        w, h = image.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            image = image.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

        # ── Depth estimation ──────────────────────────────────────────────────
        self._load_model()
        if cancellation_event.is_set():
            raise ImageTo3DError("Cancelled before depth estimation")

        try:
            result = ImageTo3DPipeline._pipe(image)
            depth_map = np.array(result["depth"], dtype=np.float32)
        except Exception as exc:
            raise ImageTo3DError(f"Depth estimation failed: {exc}") from exc

        if cancellation_event.is_set():
            raise ImageTo3DError("Cancelled after depth estimation")

        # ── Depth → mesh ──────────────────────────────────────────────────────
        try:
            vertices, faces = _depth_to_mesh(depth_map)
        except Exception as exc:
            raise ImageTo3DError(f"Mesh reconstruction failed: {exc}") from exc

        if cancellation_event.is_set():
            raise ImageTo3DError("Cancelled after mesh reconstruction")

        # ── bpy post-process → GLB ────────────────────────────────────────────
        processor = BpyPostProcessor()
        output_path = processor.process(
            np.array(vertices, dtype=np.float32),
            np.array(faces, dtype=np.int32),
            output_dir,
            name="image_mesh",
        )

        elapsed = time.time() - start
        logger.info("ImageTo3DPipeline.generate: done in %.1fs -> %s", elapsed, output_path)
        return output_path


# ── Depth → mesh helpers (no open3d) ─────────────────────────────────────────

def _depth_to_mesh(
    depth_map: np.ndarray,
    volume_depth: int = 64,
    smooth_sigma: float = 1.5,
) -> tuple[list, list]:
    """
    Convert a 2D depth map to a triangle mesh using:
      - scipy.ndimage.gaussian_filter  (smoothing)
      - skimage.measure.marching_cubes (isosurface extraction)

    Both are Python 3.13 compatible pip packages.
    """
    from scipy.ndimage import gaussian_filter
    from skimage.measure import marching_cubes

    h, w = depth_map.shape

    # Normalise depth to [0, 1]
    d_min, d_max = depth_map.min(), depth_map.max()
    if d_max - d_min < 1e-6:
        depth_norm = np.full_like(depth_map, 0.5)
    else:
        depth_norm = (depth_map - d_min) / (d_max - d_min)

    # Build a 3D occupancy volume: voxel is "filled" if depth >= its z-layer
    # Shape: (volume_depth, h, w)
    z_layers = np.linspace(0.0, 1.0, volume_depth)
    # volume[z, y, x] = 1 if depth_norm[y, x] >= z_layers[z]
    volume = (depth_norm[np.newaxis, :, :] >= z_layers[:, np.newaxis, np.newaxis]).astype(np.float32)

    # Smooth the volume to reduce staircase artifacts
    volume = gaussian_filter(volume, sigma=smooth_sigma)

    # Extract isosurface at level 0.5
    verts, faces, _normals, _values = marching_cubes(volume, level=0.5)

    # Normalise vertex coordinates to [-1, 1]
    verts = verts.astype(np.float32)
    for axis in range(3):
        lo, hi = verts[:, axis].min(), verts[:, axis].max()
        if hi - lo > 1e-6:
            verts[:, axis] = (verts[:, axis] - lo) / (hi - lo) * 2.0 - 1.0

    return verts.tolist(), faces.tolist()
