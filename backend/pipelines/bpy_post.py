"""
BpyPostProcessor: spawns bpy_worker.py as a subprocess for each operation.
This fully isolates bpy 5.1.1 from the uvicorn event loop.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Literal

import numpy as np

logger = logging.getLogger("velocity3d.bpy_post")

GLB_MAGIC = b"glTF"
ExportFormat = Literal["glb", "obj", "fbx"]

# Path to the worker script (same directory as this file)
_WORKER = str(Path(__file__).parent / "bpy_worker.py")


def _validate_glb_bytes(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == GLB_MAGIC
    except OSError:
        return False


def _python_exe() -> str:
    """
    Return the Python executable to use for the bpy worker.
    Respects HOME_VENV env var (same logic as BackendManager).
    """
    venv = os.environ.get("HOME_VENV", "")
    if venv:
        if sys.platform == "win32":
            return str(Path(venv) / "Scripts" / "python.exe")
        return str(Path(venv) / "bin" / "python3")
    return sys.executable


def _parse_worker_response(stdout: str) -> dict:
    """
    Parse the worker protocol response while tolerating stray Blender stdout.

    The worker redirects Blender stdout to stderr, but some native add-ons can
    still leak text. The protocol JSON is always the final JSON object written
    by bpy_worker.py, so scan from the end before failing.
    """
    stripped = stdout.strip()
    candidates = [stripped]
    candidates.extend(line.strip() for line in reversed(stdout.splitlines()))

    for candidate in candidates:
        if not candidate or not candidate.startswith("{"):
            continue
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    start = stripped.rfind("{")
    if start >= 0:
        try:
            return json.loads(stripped[start:])
        except json.JSONDecodeError:
            pass

    raise BpyPostProcessorError(f"bpy worker returned invalid JSON: {stdout[:200]}")


def _run_worker(task: dict, timeout: int = 300) -> str:
    """
    Spawn bpy_worker.py, send task via stdin, read result from stdout.
    Returns the output_path on success, raises BpyPostProcessorError on failure.
    """
    python = _python_exe()
    payload = json.dumps(task)

    logger.debug("bpy_worker: python=%s task_op=%s", python, task.get("op"))

    try:
        result = subprocess.run(
            [python, _WORKER],
            input=payload,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise BpyPostProcessorError(f"bpy worker timed out after {timeout}s")
    except FileNotFoundError:
        raise BpyPostProcessorError(
            f"Python executable not found: {python}\n"
            "Set HOME_VENV to your venv root in the startup screen."
        )

    # Log any stderr from the worker (bpy startup messages etc.)
    if result.stderr:
        for line in result.stderr.strip().splitlines():
            logger.debug("[bpy_worker stderr] %s", line)

    if not result.stdout.strip():
        raise BpyPostProcessorError(
            f"bpy worker produced no output (exit {result.returncode}).\n"
            f"stderr: {result.stderr[:500]}"
        )

    data = _parse_worker_response(result.stdout)

    if not data.get("ok"):
        raise BpyPostProcessorError(
            f"bpy worker failed: {data.get('error', 'unknown error')}\n"
            f"{data.get('traceback', '')}"
        )

    return data["output_path"]


class BpyPostProcessorError(Exception):
    pass


class BpyPostProcessor:
    """
    Runs bpy operations in an isolated subprocess to avoid conflicts
    between bpy 5.1.1 and the uvicorn async event loop.
    """

    def process(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        output_dir: Path,
        name: str = "generated_mesh",
    ) -> str:
        """
        Import vertices/faces into bpy, clean up mesh, export to GLB.

        Args:
            vertices: float32 array of shape (N, 3)
            faces: int32 array of shape (M, 3)
            output_dir: directory to write the output GLB
            name: mesh/object name

        Returns:
            Absolute path to the written .glb file
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(output_dir / f"{name}.glb")

        task = {
            "op": "process",
            "vertices": vertices.tolist(),
            "faces": faces.tolist(),
            "output_path": output_path,
            "name": name,
        }

        result_path = _run_worker(task)

        if not _validate_glb_bytes(result_path):
            raise BpyPostProcessorError(f"Output is not a valid GLB: {result_path}")

        logger.info("BpyPostProcessor.process: wrote %s", result_path)
        return result_path

    def export(
        self,
        source_glb: str,
        output_path: str,
        fmt: ExportFormat,
    ) -> str:
        """
        Re-export an existing GLB to glb/obj/fbx using bpy.

        Args:
            source_glb: path to the source .glb file
            output_path: destination file path
            fmt: "glb", "obj", or "fbx"

        Returns:
            output_path on success
        """
        task = {
            "op": "export",
            "source_glb": source_glb,
            "output_path": output_path,
            "format": fmt,
        }

        result_path = _run_worker(task)
        logger.info("BpyPostProcessor.export: wrote %s (%s)", result_path, fmt)
        return result_path

    def apply_texture(
        self,
        source_glb: str,
        texture_image: str,
        output_path: str,
    ) -> str:
        """
        Import an existing GLB, apply a generated texture image to mesh materials,
        and export a new textured GLB.
        """
        task = {
            "op": "apply_texture",
            "source_glb": source_glb,
            "texture_image": texture_image,
            "output_path": output_path,
        }

        result_path = _run_worker(task)
        if not _validate_glb_bytes(result_path):
            raise BpyPostProcessorError(f"Textured output is not a valid GLB: {result_path}")

        logger.info("BpyPostProcessor.apply_texture: wrote %s", result_path)
        return result_path

    def project_reference_texture(
        self,
        source_glb: str,
        reference_image: str,
        output_path: str,
    ) -> str:
        """
        Apply a user reference image as a front projection texture.

        This preserves visible source-image pixels for image-to-3D references
        and mirrors that projection to unseen back faces instead of letting the
        texture stage hallucinate a different back style.
        """
        task = {
            "op": "project_reference_texture",
            "source_glb": source_glb,
            "reference_image": reference_image,
            "output_path": output_path,
        }

        result_path = _run_worker(task)
        if not _validate_glb_bytes(result_path):
            raise BpyPostProcessorError(f"Reference textured output is not a valid GLB: {result_path}")

        logger.info("BpyPostProcessor.project_reference_texture: wrote %s", result_path)
        return result_path

    def project_reference_texture_baked(
        self,
        source_glb: str,
        albedo_image: str,
        output_path: str,
        roughness_image: str | None = None,
    ) -> str:
        """
        Apply a source-projected albedo/roughness pair using reference projection UVs.

        This is the image-to-3D viewport texture path. It avoids tiled material
        atlases and intentionally skips generated normal maps until projection
        color quality is stable.
        """
        task = {
            "op": "project_reference_texture_baked",
            "source_glb": source_glb,
            "albedo_image": albedo_image,
            "roughness_image": roughness_image,
            "output_path": output_path,
        }

        result_path = _run_worker(task)
        if not _validate_glb_bytes(result_path):
            raise BpyPostProcessorError(f"Projected textured output is not a valid GLB: {result_path}")

        logger.info("BpyPostProcessor.project_reference_texture_baked: wrote %s", result_path)
        return result_path

    def project_material_textures(
        self,
        source_glb: str,
        texture_paths: dict[str, str],
        output_path: str,
        uv_albedo_path: str | None = None,
        uv_roughness_path: str | None = None,
        uv_normal_path: str | None = None,
    ) -> str:
        """
        Apply a multi-material reference texture set.

        The worker assigns material slots by broad geometry zones and embeds the
        corresponding material textures in the exported GLB while keeping the
        source texture folder available for editing.
        """
        task = {
            "op": "project_material_textures",
            "source_glb": source_glb,
            "texture_paths": texture_paths,
            "output_path": output_path,
            "uv_albedo_path": uv_albedo_path,
            "uv_roughness_path": uv_roughness_path,
            "uv_normal_path": uv_normal_path,
        }

        result_path = _run_worker(task)
        if not _validate_glb_bytes(result_path):
            raise BpyPostProcessorError(f"Material textured output is not a valid GLB: {result_path}")

        logger.info("BpyPostProcessor.project_material_textures: wrote %s", result_path)
        return result_path

    def prepare_texture_target(
        self,
        source_glb: str,
        output_path: str,
        uv_layout_path: str,
    ) -> str:
        """
        Import a GLB, ensure UVs exist, export the UV-ready GLB, and write a
        square UV layout guide image for Stable Diffusion img2img texture synthesis.
        """
        task = {
            "op": "prepare_texture_target",
            "source_glb": source_glb,
            "output_path": output_path,
            "uv_layout_path": uv_layout_path,
        }

        result_path = _run_worker(task)
        if not _validate_glb_bytes(result_path):
            raise BpyPostProcessorError(f"UV-prepared output is not a valid GLB: {result_path}")
        if not Path(uv_layout_path).exists():
            raise BpyPostProcessorError(f"UV layout guide was not written: {uv_layout_path}")

        logger.info("BpyPostProcessor.prepare_texture_target: wrote %s and %s", result_path, uv_layout_path)
        return result_path
