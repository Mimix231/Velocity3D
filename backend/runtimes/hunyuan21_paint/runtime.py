from __future__ import annotations

import importlib
import json
import logging
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("velocity3d.hunyuan21_runtime")

RUNTIME_ID = "hunyuan3d-2.1-paint"
RUNTIME_VERSION = 4


class Hunyuan21RuntimeConfigurationError(RuntimeError):
    pass


def _base_dir() -> Path:
    configured = os.environ.get("VELOCITY_BASE_DIR", "").strip()
    if configured:
        return Path(configured)
    return Path.cwd()


def _repo_path(vendor_dir_name: str) -> Path:
    return _base_dir() / "models" / vendor_dir_name


def _runtime_root() -> Path:
    root = _base_dir() / "Runtimes"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _maybe_add_to_syspath(paths) -> None:
    for path in paths:
        if path.exists():
            path_str = str(path)
            if path_str not in sys.path:
                sys.path.insert(0, path_str)


@dataclass(frozen=True)
class Hunyuan21PaintRuntime:
    root: Path
    paint_dir: Path
    source_repo_dir: Path
    source_paint_dir: Path
    manifest_path: Path

    @property
    def real_esrgan_path(self) -> Path:
        return self.source_paint_dir / "ckpt" / "RealESRGAN_x4plus.pth"

    def activate(self) -> None:
        _maybe_add_to_syspath(
            (
                self.source_repo_dir,
                self.source_repo_dir / "hy3dshape",
                self.paint_dir,
                self.paint_dir / "utils",
                self.paint_dir / "custom_rasterizer",
                self.paint_dir / "DifferentiableRenderer",
            )
        )


def materialize_hunyuan21_paint_runtime(
    *,
    source_repo_dir: Path | None = None,
    force: bool = False,
) -> Hunyuan21PaintRuntime:
    source_repo_dir = source_repo_dir or _repo_path("hunyuan3d-2.1")
    source_paint_dir = source_repo_dir / "hy3dpaint"
    if not source_paint_dir.exists():
        raise Hunyuan21RuntimeConfigurationError(f"Hunyuan3D-2.1 paint source directory is missing: {source_paint_dir}")

    root = _runtime_root() / RUNTIME_ID
    manifest_path = root / ".velocity3d-runtime.json"
    runtime = Hunyuan21PaintRuntime(
        root=root,
        paint_dir=root / "hy3dpaint",
        source_repo_dir=source_repo_dir,
        source_paint_dir=source_paint_dir,
        manifest_path=manifest_path,
    )

    if force or not _is_runtime_current(runtime):
        _rebuild_runtime(runtime)
    else:
        _patch_runtime(runtime)

    _validate_runtime(runtime)
    return runtime


def evict_hunyuan21_runtime_modules() -> None:
    prefixes = (
        "diffusers_modules.local",
        "hunyuanpaintpbr",
        "DifferentiableRenderer",
        "custom_rasterizer",
        "utils",
    )
    exact_names = {
        "textureGenPipeline",
        "custom_rasterizer_kernel",
        "mesh_inpaint_processor",
    }
    evicted: list[str] = []
    for module_name, module in list(sys.modules.items()):
        module_file = str(getattr(module, "__file__", ""))
        if module_name in exact_names or module_name.startswith(prefixes):
            if "hunyuan" in module_file.lower() or "diffusers_modules" in module_file or module_name in exact_names:
                sys.modules.pop(module_name, None)
                evicted.append(module_name)
    if evicted:
        importlib.invalidate_caches()
        logger.info("Evicted Hunyuan3D-Paint runtime modules: %s", ", ".join(sorted(evicted)))


def _is_runtime_current(runtime: Hunyuan21PaintRuntime) -> bool:
    if not runtime.manifest_path.exists():
        return False
    try:
        manifest = json.loads(runtime.manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    if manifest.get("runtime_id") != RUNTIME_ID or manifest.get("runtime_version") != RUNTIME_VERSION:
        return False

    required_paths = (
        runtime.paint_dir / "textureGenPipeline.py",
        runtime.paint_dir / "utils" / "multiview_utils.py",
        runtime.paint_dir / "hunyuanpaintpbr" / "pipeline.py",
        runtime.paint_dir / "hunyuanpaintpbr" / "unet" / "modules.py",
        runtime.paint_dir / "DifferentiableRenderer" / "MeshRender.py",
        runtime.paint_dir / "custom_rasterizer" / "custom_rasterizer" / "render.py",
    )
    return all(path.exists() for path in required_paths)


def _rebuild_runtime(runtime: Hunyuan21PaintRuntime) -> None:
    _safe_rmtree(runtime.root)
    runtime.paint_dir.mkdir(parents=True, exist_ok=True)

    for directory_name in (
        "cfgs",
        "utils",
        "hunyuanpaintpbr",
        "DifferentiableRenderer",
        "custom_rasterizer",
    ):
        source = runtime.source_paint_dir / directory_name
        destination = runtime.paint_dir / directory_name
        if not source.exists():
            raise Hunyuan21RuntimeConfigurationError(f"Expected Hunyuan3D-Paint runtime source is missing: {source}")
        shutil.copytree(source, destination, ignore=_copy_ignore)

    for file_name in ("textureGenPipeline.py", "LICENSE", "README.md"):
        source_file = runtime.source_paint_dir / file_name
        if source_file.exists():
            shutil.copy2(source_file, runtime.paint_dir / file_name)

    _patch_runtime(runtime)
    runtime.manifest_path.write_text(
        json.dumps(
            {
                "runtime_id": RUNTIME_ID,
                "runtime_version": RUNTIME_VERSION,
                "source_repo_dir": str(runtime.source_repo_dir),
                "source_paint_dir": str(runtime.source_paint_dir),
                "execution_policy": "Velocity3D owns runtime code; downloaded Hunyuan repo is used only as assets/source input.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("Materialized Velocity3D Hunyuan3D-Paint runtime: %s", runtime.root)


def _safe_rmtree(path: Path) -> None:
    if not path.exists():
        return
    allowed_root = _runtime_root().resolve()
    resolved = path.resolve()
    if resolved.parent != allowed_root or resolved.name != RUNTIME_ID:
        raise Hunyuan21RuntimeConfigurationError(f"Refusing to remove unexpected runtime path: {path}")
    shutil.rmtree(path)


def _copy_ignore(_directory: str, names: list[str]) -> set[str]:
    ignored = {"__pycache__", ".git", ".github", "build", "dist", "*.egg-info"}
    return {name for name in names if name in ignored or name.endswith(".pyc") or name.endswith(".pyo")}


def _patch_runtime(runtime: Hunyuan21PaintRuntime) -> None:
    _patch_texture_pipeline(runtime.paint_dir / "textureGenPipeline.py")
    _patch_multiview_utils(runtime.paint_dir / "utils" / "multiview_utils.py")
    _patch_pbr_pipeline(runtime.paint_dir / "hunyuanpaintpbr" / "pipeline.py")
    _patch_pbr_modules(runtime.paint_dir / "hunyuanpaintpbr" / "unet" / "modules.py")
    _patch_attn_processor(runtime.paint_dir / "hunyuanpaintpbr" / "unet" / "attn_processor.py")
    _patch_uvwrap(runtime.paint_dir / "utils" / "uvwrap_utils.py")
    _patch_simplify(runtime.paint_dir / "utils" / "simplify_mesh_utils.py")
    _patch_mesh_utils(runtime.paint_dir / "DifferentiableRenderer" / "mesh_utils.py")
    _patch_custom_rasterizer(runtime.paint_dir / "custom_rasterizer")


def _patch_texture_pipeline(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    patched = content.replace(
        "for i in range(len(enhance_images)):",
        "for i in range(len(enhance_images[\"albedo\"])):",
    )
    _write_if_changed(path, content, patched)


def _patch_multiview_utils(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    patched = content.replace(
        "        custom_pipeline = os.path.join(os.path.dirname(__file__),\"..\",\"hunyuanpaintpbr\")\n",
        "        custom_pipeline = getattr(config, \"custom_pipeline\", None) or os.path.join(os.path.dirname(__file__), \"..\", \"hunyuanpaintpbr\")\n",
    )
    patched = patched.replace(
        "            custom_pipeline=custom_pipeline, \n",
        "            custom_pipeline=str(custom_pipeline),\n",
    )
    _write_if_changed(path, content, patched)


def _patch_pbr_pipeline(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    patched = content
    if "vae_param = next(self.vae.parameters())" not in patched:
        patched = patched.replace(
            "        dtype = next(self.vae.parameters()).dtype\n"
            "        images = (images - 0.5) * 2.0\n"
            "        posterior = self.vae.encode(images.to(dtype)).latent_dist\n",
            "        vae_param = next(self.vae.parameters())\n"
            "        dtype = vae_param.dtype\n"
            "        images = (images - 0.5) * 2.0\n"
            "        images = images.to(device=vae_param.device, dtype=dtype)\n"
            "        posterior = self.vae.encode(images).latent_dist\n",
        )
    patched = patched.replace(
        "        images_vae = images_vae.to(device=self.vae.device, dtype=self.unet.dtype)\n",
        "        images_vae = images_vae.to(device=next(self.vae.parameters()).device, dtype=self.unet.dtype)\n",
    )
    if "target_device = next(self.vae.parameters()).device" not in patched:
        patched = patched.replace(
            "        def convert_pil_list_to_tensor(images):\n"
            "            bg_c = [1.0, 1.0, 1.0]\n",
            "        def convert_pil_list_to_tensor(images):\n"
            "            target_device = next(self.vae.parameters()).device\n"
            "            bg_c = [1.0, 1.0, 1.0]\n",
        )
    patched = patched.replace(
        '                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous().half().to("cuda")\n',
        "                    img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).contiguous()\n"
        "                    img = img.to(device=target_device, dtype=self.unet.dtype)\n",
    )
    if "target_device = next(self.unet.parameters()).device" not in patched:
        patched = patched.replace(
            "        if guidance_scale > 1:\n",
            "        target_device = next(self.unet.parameters()).device\n"
            "        target_dtype = next(self.unet.parameters()).dtype\n"
            "        prompt_embeds = prompt_embeds.to(device=target_device, dtype=target_dtype)\n"
            "        negative_prompt_embeds = negative_prompt_embeds.to(device=target_device, dtype=target_dtype)\n"
            "\n"
            "        if guidance_scale > 1:\n",
        )
    _write_if_changed(path, content, patched)


def _patch_pbr_modules(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    patched = content.replace(
        "        else:\n"
        "            batch_size = 1\n"
        "            dino_proceesed_images = self.dino_processor(images=images, return_tensors=\"pt\").pixel_values\n"
        "            dino_proceesed_images = torch.stack(\n"
        "                [torch.from_numpy(np.array(image)) for image in dino_proceesed_images], dim=0\n"
        "            )\n"
        "        dino_param = next(self.dino_v2.parameters())\n"
        "        dino_proceesed_images = dino_proceesed_images.to(dino_param)\n",
        "        else:\n"
        "            batch_size = 1\n"
        "            dino_proceesed_images = self.dino_processor(images=images, return_tensors=\"pt\").pixel_values\n"
        "        dino_param = next(self.dino_v2.parameters())\n"
        "        dino_proceesed_images = dino_proceesed_images.to(device=dino_param.device, dtype=dino_param.dtype)\n",
    ).replace(
        "        dino_hidden_states = rearrange(dino_hidden_states.to(dino_param), \"(b n) l c -> b (n l) c\", b=batch_size)\n",
        "        dino_hidden_states = rearrange(\n"
        "            dino_hidden_states.to(device=dino_param.device, dtype=dino_param.dtype),\n"
        "            \"(b n) l c -> b (n l) c\",\n"
        "            b=batch_size,\n"
        "        )\n",
    )
    patched = _patch_attention_block_source(patched)
    _write_if_changed(path, content, patched)


def _patch_attention_block_source(source: str) -> str:
    patched = source
    if "def _velocity3d_ensure_mda_processor" not in patched:
        patched = patched.replace(
            "    def __getattr__(self, name: str):\n",
            "    def _velocity3d_ensure_mda_processor(self):\n"
            "        if not self.use_mda:\n"
            "            return\n"
            "        if self.attn1.processor.__class__.__name__ == \"SelfAttnProcessor2_0\":\n"
            "            return\n"
            "\n"
            "        self.attn1.set_processor(\n"
            "            SelfAttnProcessor2_0(\n"
            "                query_dim=self.dim,\n"
            "                heads=self.num_attention_heads,\n"
            "                dim_head=self.attention_head_dim,\n"
            "                dropout=self.dropout,\n"
            "                bias=self.attention_bias,\n"
            "                cross_attention_dim=None,\n"
            "                upcast_attention=self.attn1.upcast_attention,\n"
            "                out_bias=True,\n"
            "                pbr_setting=self.pbr_setting,\n"
            "            )\n"
            "        )\n"
            "        for token in self.pbr_setting:\n"
            "            if token == \"albedo\":\n"
            "                continue\n"
            "            getattr(self.attn1.processor, f\"to_q_{token}\").load_state_dict(self.attn1.to_q.state_dict())\n"
            "            getattr(self.attn1.processor, f\"to_k_{token}\").load_state_dict(self.attn1.to_k.state_dict())\n"
            "            getattr(self.attn1.processor, f\"to_v_{token}\").load_state_dict(self.attn1.to_v.state_dict())\n"
            "            getattr(self.attn1.processor, f\"to_out_{token}\").load_state_dict(self.attn1.to_out.state_dict())\n"
            "        self.attn1.processor.to(device=self.attn1.to_q.weight.device, dtype=self.attn1.to_q.weight.dtype)\n"
            "\n"
            "    def __getattr__(self, name: str):\n",
        )
    if "def _velocity3d_flatten_pbr_attention" not in patched:
        patched = patched.replace(
            "    def __getattr__(self, name: str):\n",
            "    def _velocity3d_flatten_pbr_attention(self, attn_output, num_in_batch, n_pbr, flat_batch_size, label):\n"
            "        if attn_output.ndim == 5:\n"
            "            return rearrange(attn_output, \"b n_pbr n l c -> (b n_pbr n) l c\", n=num_in_batch, n_pbr=n_pbr)\n"
            "\n"
            "        if attn_output.ndim == 4:\n"
            "            if attn_output.shape[1] == n_pbr:\n"
            "                return rearrange(attn_output, \"b n_pbr (n l) c -> (b n_pbr n) l c\", n=num_in_batch, n_pbr=n_pbr)\n"
            "            if attn_output.shape[1] == num_in_batch:\n"
            "                attn_output = attn_output.unsqueeze(1).repeat(1, n_pbr, 1, 1, 1)\n"
            "                return rearrange(attn_output, \"b n_pbr n l c -> (b n_pbr n) l c\")\n"
            "            if attn_output.shape[0] % n_pbr == 0:\n"
            "                return rearrange(attn_output, \"(b n_pbr) n l c -> (b n_pbr n) l c\", n_pbr=n_pbr)\n"
            "\n"
            "        if attn_output.ndim == 3:\n"
            "            if attn_output.shape[0] == flat_batch_size:\n"
            "                return attn_output\n"
            "            base_batch = max(1, flat_batch_size // max(1, n_pbr * num_in_batch))\n"
            "            if attn_output.shape[0] == base_batch and attn_output.shape[1] % num_in_batch == 0:\n"
            "                attn_output = rearrange(attn_output, \"b (n l) c -> b n l c\", n=num_in_batch)\n"
            "                attn_output = attn_output.unsqueeze(1).repeat(1, n_pbr, 1, 1, 1)\n"
            "                return rearrange(attn_output, \"b n_pbr n l c -> (b n_pbr n) l c\")\n"
            "            if attn_output.shape[0] == base_batch * n_pbr and attn_output.shape[1] % num_in_batch == 0:\n"
            "                return rearrange(attn_output, \"(b n_pbr) (n l) c -> (b n_pbr n) l c\", n_pbr=n_pbr, n=num_in_batch)\n"
            "\n"
            "        raise ValueError(\n"
            "            f\"Velocity3D could not normalize {label} attention output with shape {tuple(attn_output.shape)} \"\n"
            "            f\"for num_in_batch={num_in_batch}, n_pbr={n_pbr}, flat_batch_size={flat_batch_size}.\"\n"
            "        )\n"
            "\n"
            "    def __getattr__(self, name: str):\n",
        )
    if "self._velocity3d_ensure_mda_processor()" not in patched:
        patched = patched.replace(
            "        if self.use_mda:\n"
            "            mda_norm_hidden_states = rearrange(\n",
            "        if self.use_mda:\n"
            "            self._velocity3d_ensure_mda_processor()\n"
            "            mda_norm_hidden_states = rearrange(\n",
        )
    if '"material self"' not in patched:
        patched = patched.replace(
            '            attn_output = rearrange(attn_output, "b n_pbr n l c -> (b n_pbr n) l c")\n',
            '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
            '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "material self"\n'
            '            )\n',
            1,
        )
    if '"reference"' not in patched:
        patched = patched.replace(
            '            if attn_output.ndim == 3:\n'
            '                attn_output = rearrange(attn_output, "b (n l) c -> b n l c", n=num_in_batch)\n'
            '                attn_output = attn_output.unsqueeze(1).repeat(1, N_pbr, 1, 1, 1)\n'
            '            elif attn_output.ndim == 4:\n'
            '                attn_output = rearrange(attn_output, "b n_pbr (n l) c -> b n_pbr n l c", n=num_in_batch, n_pbr=N_pbr)\n'
            '            attn_output = rearrange(attn_output, "b n_pbr n l c -> (b n_pbr n) l c")\n',
            '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
            '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "reference"\n'
            '            )\n',
        )
        patched = patched.replace(
            '            attn_output = rearrange(attn_output, "b n_pbr (n l) c -> (b n_pbr n) l c", n=num_in_batch, n_pbr=N_pbr)\n',
            '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
            '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "reference"\n'
            '            )\n',
        )
    if '"multiview"' not in patched:
        patched = patched.replace(
            '            attn_output = rearrange(attn_output, "(b n_pbr) (n l) c -> (b n_pbr n) l c", n_pbr=N_pbr, n=num_in_batch)\n',
            '            attn_output = self._velocity3d_flatten_pbr_attention(\n'
            '                attn_output, num_in_batch, N_pbr, hidden_states.shape[0], "multiview"\n'
            '            )\n',
        )
    return patched


def _patch_attn_processor(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    marker = "target_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states"
    patched = content
    if marker not in patched:
        patched = patched.replace(
            "        batch_size, sequence_length, _ = (\n"
            "            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape\n"
            "        )\n",
            "        target_states = hidden_states if encoder_hidden_states is None else encoder_hidden_states\n"
            "        if target_states.ndim > 3:\n"
            "            target_states = target_states.reshape(-1, target_states.shape[-2], target_states.shape[-1])\n"
            "            if encoder_hidden_states is None:\n"
            "                hidden_states = target_states\n"
            "            else:\n"
            "                encoder_hidden_states = target_states\n"
            "        batch_size, sequence_length, _ = target_states.shape\n",
        )
    _write_if_changed(path, content, patched)


def _patch_uvwrap(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    marker = "atlas_result = xatlas.parametrize(mesh.vertices, mesh.faces)"
    patched = content
    if marker not in patched:
        patched = patched.replace(
            "    vmapping, indices, uvs = xatlas.parametrize(mesh.vertices, mesh.faces)\n",
            "    atlas_result = xatlas.parametrize(mesh.vertices, mesh.faces)\n"
            "    if len(atlas_result) < 3:\n"
            "        raise ValueError(f\"xatlas.parametrize returned {len(atlas_result)} values; expected at least 3.\")\n"
            "    vmapping, indices, uvs = atlas_result[:3]\n",
        )
    _write_if_changed(path, content, patched)


def _patch_simplify(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    patched = content.replace(
        "courent = courent.simplify_quadric_decimation(target_count)",
        "courent = courent.simplify_quadric_decimation(face_count=target_count)",
    )
    _write_if_changed(path, content, patched)


def _patch_mesh_utils(path: Path) -> None:
    if not path.exists():
        return
    content = path.read_text(encoding="utf-8")
    patched = content
    if "_BPY_IMPORT_ERROR" not in patched:
        patched = patched.replace(
            "import bpy\n",
            "try:\n"
            "    import bpy\n"
            "    _BPY_IMPORT_ERROR = None\n"
            "except Exception as exc:\n"
            "    bpy = None\n"
            "    _BPY_IMPORT_ERROR = exc\n",
        )
    if "Blender bpy module is unavailable" not in patched:
        patched = patched.replace(
            "def convert_obj_to_glb(\n"
            "    obj_path: str,\n"
            "    glb_path: str,\n"
            "    shade_type: str = \"SMOOTH\",\n"
            "    auto_smooth_angle: float = 60,\n"
            "    merge_vertices: bool = False,\n"
            ") -> bool:\n"
            "    \"\"\"Convert OBJ file to GLB format using Blender.\"\"\"\n"
            "    try:\n",
            "def convert_obj_to_glb(\n"
            "    obj_path: str,\n"
            "    glb_path: str,\n"
            "    shade_type: str = \"SMOOTH\",\n"
            "    auto_smooth_angle: float = 60,\n"
            "    merge_vertices: bool = False,\n"
            ") -> bool:\n"
            "    \"\"\"Convert OBJ file to GLB format using Blender.\"\"\"\n"
            "    if bpy is None:\n"
            "        raise ImportError(f\"Blender bpy module is unavailable: {_BPY_IMPORT_ERROR}\")\n"
            "    try:\n",
        )
    _write_if_changed(path, content, patched)


def _patch_custom_rasterizer(custom_rasterizer_dir: Path) -> None:
    render_path = custom_rasterizer_dir / "custom_rasterizer" / "render.py"
    if render_path.exists():
        content = render_path.read_text(encoding="utf-8")
        patched = content
        if "tri = tri.to(device=pos.device, dtype=torch.int32).contiguous()" not in patched:
            patched = patched.replace(
                "def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):\n"
                "    assert pos.device == tri.device\n"
                "    findices, barycentric = custom_rasterizer_kernel.rasterize_image(\n",
                "def rasterize(pos, tri, resolution, clamp_depth=torch.zeros(0), use_depth_prior=0):\n"
                "    assert pos.device == tri.device\n"
                "    tri = tri.to(device=pos.device, dtype=torch.int32).contiguous()\n"
                "    if clamp_depth.numel() > 0:\n"
                "        clamp_depth = clamp_depth.to(device=pos.device, dtype=torch.float32).contiguous()\n"
                "    findices, barycentric = custom_rasterizer_kernel.rasterize_image(\n",
            )
        _write_if_changed(render_path, content, patched)

    for rasterizer_source in (
        custom_rasterizer_dir / "lib" / "custom_rasterizer_kernel_for_windows" / "rasterizer_gpu.cu",
        custom_rasterizer_dir / "lib" / "custom_rasterizer_kernel" / "rasterizer_gpu.cu",
    ):
        if not rasterizer_source.exists():
            continue
        content = rasterizer_source.read_text(encoding="utf-8")
        patched = content.replace(
            "uint64_t maxint = (uint64_t)MAXINT * (uint64_t)MAXINT + (MAXINT - 1);",
            "int64_t maxint = (int64_t)MAXINT * (int64_t)MAXINT + (MAXINT - 1);",
        ).replace(
            "auto z_min = torch::ones({height, width}, INT64_options) * (uint64_t)maxint;",
            "auto z_min = torch::ones({height, width}, INT64_options) * maxint;",
        ).replace(
            "auto z_min = torch::ones({height, width}, INT64_options) * (long)maxint;",
            "auto z_min = torch::ones({height, width}, INT64_options) * maxint;",
        ).replace(
            "auto z_min = torch::ones({height, width}, UINT64_options) * (uint64_t)maxint;",
            "auto z_min = torch::ones({height, width}, INT64_options) * maxint;",
        ).replace(
            "auto UINT64_options = torch::TensorOptions().dtype(torch::kUInt64).device(torch::kCUDA, device_id).requires_grad(false);",
            "auto INT64_options = torch::TensorOptions().dtype(torch::kInt64).device(torch::kCUDA, device_id).requires_grad(false);",
        ).replace(
            "z_min.data_ptr<uint64_t>()",
            "z_min.data_ptr<int64_t>()",
        )
        _write_if_changed(rasterizer_source, content, patched)

    obsolete_shim = custom_rasterizer_dir / "custom_rasterizer_kernel.py"
    if obsolete_shim.exists():
        obsolete_shim.unlink()


def _validate_runtime(runtime: Hunyuan21PaintRuntime) -> None:
    modules_path = runtime.paint_dir / "hunyuanpaintpbr" / "unet" / "modules.py"
    if not modules_path.exists():
        raise Hunyuan21RuntimeConfigurationError(f"Velocity3D Hunyuan paint runtime is incomplete: {modules_path}")
    modules_source = modules_path.read_text(encoding="utf-8")
    required_markers = (
        "def _velocity3d_flatten_pbr_attention",
        '"material self"',
        '"reference"',
        '"multiview"',
    )
    missing = [marker for marker in required_markers if marker not in modules_source]
    if missing:
        raise Hunyuan21RuntimeConfigurationError(
            f"Velocity3D Hunyuan paint runtime is missing required patches: {', '.join(missing)}"
        )

    pyd_candidates = (
        runtime.paint_dir / "custom_rasterizer" / "custom_rasterizer_kernel.cp313-win_amd64.pyd",
        runtime.paint_dir / "custom_rasterizer" / "custom_rasterizer_kernel.cp312-win_amd64.pyd",
        runtime.paint_dir / "DifferentiableRenderer" / "mesh_inpaint_processor.cp313-win_amd64.pyd",
        runtime.paint_dir / "DifferentiableRenderer" / "mesh_inpaint_processor.cp312-win_amd64.pyd",
    )
    if sys.platform == "win32" and not any(path.exists() for path in pyd_candidates):
        logger.warning(
            "Velocity3D Hunyuan paint runtime was materialized without native .pyd outputs. "
            "Run Install / Download to build custom_rasterizer and mesh_inpaint_processor."
        )


def _write_if_changed(path: Path, before: str, after: str) -> None:
    if after != before:
        path.write_text(after, encoding="utf-8")
