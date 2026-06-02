from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class GenerationPreset:
    id: str
    label: str
    description: str
    target_face_count: int
    shap_e_steps: int
    shap_e_guidance: float
    image_max_dim: int
    image_volume_depth: int
    image_smooth_sigma: float
    trellis_simplify: float
    trellis_texture_size: int
    trellis2_decimation_target: int
    trellis2_texture_size: int
    hunyuan_paint_profile: str
    hunyuan_remesh_target: int


PRESETS: dict[str, GenerationPreset] = {
    "preview": GenerationPreset(
        id="preview",
        label="Preview",
        description="Fast iteration with lighter meshes and smaller textures.",
        target_face_count=25000,
        shap_e_steps=36,
        shap_e_guidance=12.0,
        image_max_dim=384,
        image_volume_depth=48,
        image_smooth_sigma=1.7,
        trellis_simplify=0.975,
        trellis_texture_size=1024,
        trellis2_decimation_target=50000,
        trellis2_texture_size=1024,
        hunyuan_paint_profile="fast",
        hunyuan_remesh_target=25000,
    ),
    "balanced": GenerationPreset(
        id="balanced",
        label="Balanced",
        description="General-purpose quality and speed for local generation.",
        target_face_count=80000,
        shap_e_steps=64,
        shap_e_guidance=15.0,
        image_max_dim=512,
        image_volume_depth=64,
        image_smooth_sigma=1.5,
        trellis_simplify=0.95,
        trellis_texture_size=2048,
        trellis2_decimation_target=100000,
        trellis2_texture_size=2048,
        hunyuan_paint_profile="balanced",
        hunyuan_remesh_target=50000,
    ),
    "building_module": GenerationPreset(
        id="building_module",
        label="Building Module",
        description="Lower-poly, sharper hard-surface output for modular game/building kits.",
        target_face_count=35000,
        shap_e_steps=48,
        shap_e_guidance=14.0,
        image_max_dim=512,
        image_volume_depth=56,
        image_smooth_sigma=1.25,
        trellis_simplify=0.985,
        trellis_texture_size=1024,
        trellis2_decimation_target=35000,
        trellis2_texture_size=1024,
        hunyuan_paint_profile="fast",
        hunyuan_remesh_target=35000,
    ),
    "game_asset": GenerationPreset(
        id="game_asset",
        label="Game Asset",
        description="Game-ready mesh density with room for texture detail.",
        target_face_count=50000,
        shap_e_steps=56,
        shap_e_guidance=14.5,
        image_max_dim=512,
        image_volume_depth=64,
        image_smooth_sigma=1.35,
        trellis_simplify=0.975,
        trellis_texture_size=2048,
        trellis2_decimation_target=50000,
        trellis2_texture_size=2048,
        hunyuan_paint_profile="balanced",
        hunyuan_remesh_target=50000,
    ),
    "production": GenerationPreset(
        id="production",
        label="Production",
        description="Higher mesh and texture budgets for final-quality exports.",
        target_face_count=200000,
        shap_e_steps=80,
        shap_e_guidance=16.0,
        image_max_dim=768,
        image_volume_depth=96,
        image_smooth_sigma=1.2,
        trellis_simplify=0.90,
        trellis_texture_size=2048,
        trellis2_decimation_target=250000,
        trellis2_texture_size=4096,
        hunyuan_paint_profile="quality",
        hunyuan_remesh_target=120000,
    ),
}


def _get_option(options: Any, name: str, default: Any = None) -> Any:
    if options is None:
        return default
    if isinstance(options, dict):
        return options.get(name, default)
    return getattr(options, name, default)


def resolve_generation_preset(options: Any = None) -> GenerationPreset:
    preset_id = str(_get_option(options, "preset", "balanced") or "balanced")
    base = PRESETS.get(preset_id, PRESETS["balanced"])

    target_face_count = _get_option(options, "target_face_count", None)
    texture_size = _get_option(options, "texture_size", None)

    if target_face_count is None and texture_size is None:
        return base

    next_target = int(target_face_count) if target_face_count is not None else base.target_face_count
    next_texture = int(texture_size) if texture_size is not None else base.trellis_texture_size

    next_target = max(1000, min(2_000_000, next_target))
    next_texture = max(512, min(4096, next_texture))

    return GenerationPreset(
        **{
            **base.__dict__,
            "target_face_count": next_target,
            "trellis_texture_size": next_texture,
            "trellis2_texture_size": next_texture,
            "hunyuan_remesh_target": next_target,
        }
    )
