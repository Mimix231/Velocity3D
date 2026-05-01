from __future__ import annotations

import io
import json
import logging
import random
import statistics
import threading
from pathlib import Path

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps, ImageStat

from backend.providers.base import ProviderDependencyError, ProviderExecutionError
from backend.providers.helpers import checkpoints_root, configure_huggingface_cache, safe_stem

DEFAULT_TEXTURE_CHECKPOINT = "stabilityai/stable-diffusion-xl-base-1.0"
logger = logging.getLogger("velocity3d.texture")
DEFAULT_NEGATIVE_PROMPT = (
    "blurry, low quality, low resolution, watermark, text, logo, full object render, character render, "
    "perspective view, scene background, sky, floor, drop shadow, strong lighting, baked highlights, "
    "baked shadows, photography, camera, frame, border, uv guide, wireframe, white mask, black and white, "
    "monochrome, grayscale, blank white texture, silhouette mask, empty texture, uncolored atlas"
)


def build_texture_prompt(source_prompt: str | None, uv_guided: bool = False) -> str:
    cleaned = (source_prompt or "").strip()
    prefix = (
        "single square 3D model albedo texture atlas, richly colored material texture sheet, "
        "game-ready diffuse map, hand-authored online 3D asset texture sheet, trim sheets, panels, "
        "painted material color variation, clean surface detail, no lighting, no shadows, no perspective, "
        "no complete object render"
    )
    if uv_guided:
        prefix = (
            "colored 3D model UV texture atlas, richly colored albedo map, hand-painted material islands, "
            "game-ready diffuse texture sheet, trim details, panels, edges, material wear, decorative color blocks, "
            "online 3D asset texture, no lighting, no shadows, no black-and-white UV mask, no blank white islands"
        )

    if cleaned:
        return f"{prefix}, {cleaned}"
    return prefix


def build_material_texture_prompt(material: str, source_prompt: str | None) -> str:
    cleaned = (source_prompt or "").strip()
    material_prompts = {
        "roof": (
            "seamless terracotta roof tile albedo texture, rows of curved clay tiles, baked color variation, "
            "game asset material map, no object render, no perspective, no shadows"
        ),
        "wall": (
            "painted plaster wall albedo texture for a 3D building asset, subtle stucco grain, warm color variation, "
            "flat material map, no windows, no complete building, no perspective, no shadows"
        ),
        "trim": (
            "cream painted architectural trim albedo texture, bevel edge wear, fascia and window frame material, "
            "flat game asset material map, no perspective, no object render"
        ),
        "window": (
            "dark glass and painted window frame albedo texture, rectangular pane details, subtle reflections baked as color only, "
            "flat material map, no perspective, no full object"
        ),
        "misc": (
            "cohesive secondary material albedo texture for a stylized 3D asset, weathered painted surfaces, "
            "flat texture sheet, no perspective, no full object"
        ),
    }
    prompt = material_prompts.get(material, material_prompts["misc"])
    if cleaned:
        return f"{prompt}, source reference style: {cleaned}"
    return prompt


def _palette_for_prompt(prompt: str | None) -> tuple[str, str]:
    text = (prompt or "").lower()
    if any(word in text for word in ("wood", "timber", "house", "cabin", "fantasy", "medieval")):
        return "#5b321d", "#d49a4a"
    if any(word in text for word in ("stone", "rock", "castle", "ruin", "wall")):
        return "#314147", "#b4a07a"
    if any(word in text for word in ("metal", "robot", "mech", "armor", "sci-fi", "scifi")):
        return "#26364b", "#b8c7d9"
    if any(word in text for word in ("plant", "tree", "forest", "nature", "moss")):
        return "#244425", "#8abf57"
    return "#3d2f66", "#d79652"


def _mean_saturation(image: Image.Image) -> float:
    hsv = image.convert("HSV")
    stat = ImageStat.Stat(hsv)
    return float(stat.mean[1])


def _ensure_colored_albedo(image: Image.Image, prompt: str | None) -> Image.Image:
    """
    Guard against SD reproducing the UV guide as a black/white mask.

    The texture pass must produce a colored albedo map. If the generated image is
    nearly monochrome, colorize the luminance with a prompt-derived palette and
    lift saturation so the final mesh does not render with a white mask texture.
    """
    rgb = image.convert("RGB")
    if _mean_saturation(rgb) >= 34:
        return ImageEnhance.Color(rgb).enhance(1.18)

    dark, light = _palette_for_prompt(prompt)
    colorized = ImageOps.colorize(rgb.convert("L"), black=dark, white=light)
    colorized = ImageEnhance.Color(colorized).enhance(1.35)
    return ImageEnhance.Contrast(colorized).enhance(1.08)


def _uv_mask_from_layout(uv_layout_path: Path, size: tuple[int, int]) -> Image.Image:
    layout = Image.open(uv_layout_path).convert("L").resize(size, Image.Resampling.LANCZOS)
    mask = layout.point(lambda value: 255 if value > 16 else 0, "L")
    return mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.GaussianBlur(0.45))


def _apply_uv_mask(texture: Image.Image, uv_layout_path: Path | None, prompt: str | None) -> Image.Image:
    colored = _ensure_colored_albedo(texture, prompt).resize((1024, 1024), Image.Resampling.LANCZOS)
    if uv_layout_path is None or not uv_layout_path.exists():
        return colored

    mask = _uv_mask_from_layout(uv_layout_path, colored.size)
    atlas = Image.new("RGB", colored.size, "black")
    atlas.paste(colored, (0, 0), mask)
    return atlas


def _expanded_bbox(bbox: tuple[int, int, int, int], size: tuple[int, int], padding_ratio: float = 0.025) -> tuple[int, int, int, int]:
    left, top, right, bottom = bbox
    width, height = size
    pad_x = max(2, int((right - left) * padding_ratio))
    pad_y = max(2, int((bottom - top) * padding_ratio))
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def _reference_subject_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    alpha = image.getchannel("A")
    alpha_min, alpha_max = alpha.getextrema()
    alpha_bbox = alpha.point(lambda value: 255 if value > 8 else 0, "L").getbbox()
    if alpha_min < 250 and alpha_bbox:
        return _expanded_bbox(alpha_bbox, image.size)

    rgb = image.convert("RGB")
    width, height = rgb.size
    stride = max(1, max(width, height) // 220)
    border: list[tuple[int, int, int]] = []
    for x in range(0, width, stride):
        border.append(rgb.getpixel((x, 0)))
        border.append(rgb.getpixel((x, height - 1)))
    for y in range(0, height, stride):
        border.append(rgb.getpixel((0, y)))
        border.append(rgb.getpixel((width - 1, y)))

    bg = (
        int(statistics.median(pixel[0] for pixel in border)),
        int(statistics.median(pixel[1] for pixel in border)),
        int(statistics.median(pixel[2] for pixel in border)),
    )
    bg_luma = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    threshold_sq = 72 * 72 if bg_luma > 65 else 54 * 54

    mask = Image.new("L", rgb.size, 0)
    mask_data = []
    for red, green, blue in rgb.getdata():
        dr = red - bg[0]
        dg = green - bg[1]
        db = blue - bg[2]
        value = max(red, green, blue)
        chroma = value - min(red, green, blue)
        luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        far_from_background = dr * dr + dg * dg + db * db > threshold_sq
        visible_dark_detail = bg_luma < 75 and luma > bg_luma + 22 and chroma > 12
        mask_data.append(255 if far_from_background or visible_dark_detail else 0)

    mask.putdata(mask_data)
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3))
    color_bbox = mask.getbbox()
    if color_bbox:
        return _expanded_bbox(color_bbox, image.size, padding_ratio=0.018)

    luminance = image.convert("L")
    luminance_bbox = luminance.point(lambda value: 255 if value > max(30, bg_luma + 20) else 0, "L").getbbox()
    if luminance_bbox:
        return _expanded_bbox(luminance_bbox, image.size, padding_ratio=0.018)

    return None


def _average_opaque_color(image: Image.Image) -> tuple[int, int, int]:
    alpha = image.getchannel("A").point(lambda value: 255 if value > 20 else 0, "L")
    rgb = image.convert("RGB")
    try:
        mean = ImageStat.Stat(rgb, mask=alpha).mean
    except Exception:
        mean = ImageStat.Stat(rgb).mean

    if len(mean) < 3 or sum(mean) <= 1:
        return (126, 92, 58)
    return tuple(max(0, min(255, int(channel))) for channel in mean[:3])


def _strict_reference_cutout(image: Image.Image) -> Image.Image:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_min, _alpha_max = alpha.getextrema()
    if alpha_min < 250:
        return rgba

    rgb = rgba.convert("RGB")
    width, height = rgb.size
    stride = max(1, max(width, height) // 220)
    border: list[tuple[int, int, int]] = []
    for x in range(0, width, stride):
        border.append(rgb.getpixel((x, 0)))
        border.append(rgb.getpixel((x, height - 1)))
    for y in range(0, height, stride):
        border.append(rgb.getpixel((0, y)))
        border.append(rgb.getpixel((width - 1, y)))

    bg = (
        int(statistics.median(pixel[0] for pixel in border)),
        int(statistics.median(pixel[1] for pixel in border)),
        int(statistics.median(pixel[2] for pixel in border)),
    )
    bg_luma = 0.2126 * bg[0] + 0.7152 * bg[1] + 0.0722 * bg[2]
    threshold_sq = 60 * 60 if bg_luma > 65 else 42 * 42

    mask = Image.new("L", rgb.size, 0)
    mask_data = []
    for red, green, blue in rgb.getdata():
        dr = red - bg[0]
        dg = green - bg[1]
        db = blue - bg[2]
        value = max(red, green, blue)
        chroma = value - min(red, green, blue)
        luma = 0.2126 * red + 0.7152 * green + 0.0722 * blue
        selected = dr * dr + dg * dg + db * db > threshold_sq or (
            bg_luma < 75 and luma > bg_luma + 18 and chroma > 10
        )
        mask_data.append(255 if selected else 0)

    mask.putdata(mask_data)
    mask = mask.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(0.45))
    rgba.putalpha(mask)
    return rgba


def prepare_reference_texture_image(reference_image_bytes: bytes, output_path: Path, max_size: int = 1536) -> Path:
    """
    Prepare a user reference image for front-projected texture baking.

    The image-to-3D input is usually a background-removed cutout. This crops the
    visible subject, flattens transparency to a subject-derived average color,
    and keeps the original pixels as the texture source instead of asking SD to
    repaint the whole object.
    """
    try:
        image = Image.open(io.BytesIO(reference_image_bytes)).convert("RGBA")
    except Exception as exc:
        raise ProviderExecutionError(f"Could not decode reference texture image: {exc}") from exc

    bbox = _reference_subject_bbox(image)
    if bbox:
        image = image.crop(bbox)
    image = _strict_reference_cutout(image)

    background = Image.new("RGB", image.size, _average_opaque_color(image))
    background.paste(image.convert("RGB"), (0, 0), image.getchannel("A"))
    background = ImageEnhance.Color(background).enhance(1.08)
    background.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    background.save(output_path)
    return output_path


def _roughness_from_reference_albedo(albedo: Image.Image) -> Image.Image:
    gray = ImageOps.grayscale(albedo.convert("RGB"))
    gray = ImageOps.autocontrast(gray)
    # Keep generated reference textures matte. This avoids the wet/metallic
    # look from fake normals/specular while preserving enough surface variation.
    roughness = gray.point(lambda value: max(142, min(238, int(212 - (value - 128) * 0.12))), "L")
    return ImageEnhance.Contrast(roughness).enhance(0.55)


def _extension_fill_from_albedo(albedo: Image.Image, mode: str) -> Image.Image:
    rgb = albedo.convert("RGB")
    if mode == "back":
        extended = ImageOps.mirror(rgb)
    else:
        extended = rgb.filter(ImageFilter.GaussianBlur(10))
        extended = ImageEnhance.Contrast(extended).enhance(0.82)

    return ImageEnhance.Color(extended).enhance(0.92)


def build_projected_texture_set(reference_image_bytes: bytes, output_dir: Path, max_size: int = 1536) -> dict[str, Path]:
    """
    Build the image-to-3D viewport texture set from the reference itself.

    This path deliberately avoids the material-tile SD atlas. The model gets a
    source-projected albedo plus a derived roughness map, while side/back fill
    images are kept in the texture folder for the next reconstruction stage.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    albedo_path = prepare_reference_texture_image(
        reference_image_bytes,
        output_dir / "projected_albedo.png",
        max_size=max_size,
    )

    albedo = Image.open(albedo_path).convert("RGB")
    roughness_path = output_dir / "projected_roughness.png"
    back_fill_path = output_dir / "back_fill_albedo.png"
    side_fill_path = output_dir / "side_fill_albedo.png"

    _roughness_from_reference_albedo(albedo).save(roughness_path)
    _extension_fill_from_albedo(albedo, "back").save(back_fill_path)
    _extension_fill_from_albedo(albedo, "side").save(side_fill_path)

    manifest_path = output_dir / "manifest.json"
    textures = {
        "projected_albedo": albedo_path,
        "projected_roughness": roughness_path,
        "back_fill_albedo": back_fill_path,
        "side_fill_albedo": side_fill_path,
    }
    manifest_path.write_text(
        json.dumps(
            {
                "type": "velocity3d_projected_reference_texture_set",
                "version": 1,
                "source": "background_removed_reference_projection",
                "textures": {name: path.name for name, path in textures.items()},
                "assignment": {
                    "projected_albedo": "packed into viewport GLB using reference projection UVs",
                    "projected_roughness": "packed into viewport GLB; normal map intentionally disabled",
                    "back_fill_albedo": "reserved for later side/back reconstruction",
                    "side_fill_albedo": "reserved for later side/back reconstruction",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return textures


def _load_prepared_reference(reference_image_bytes: bytes, max_size: int = 1536) -> Image.Image:
    try:
        image = Image.open(io.BytesIO(reference_image_bytes)).convert("RGBA")
    except Exception as exc:
        raise ProviderExecutionError(f"Could not decode reference texture image: {exc}") from exc

    bbox = _reference_subject_bbox(image)
    if bbox:
        image = image.crop(bbox)
    image = _strict_reference_cutout(image)

    background = Image.new("RGB", image.size, _average_opaque_color(image))
    background.paste(image.convert("RGB"), (0, 0), image.getchannel("A"))
    background = ImageEnhance.Color(background).enhance(1.08)
    background.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
    return background


def _material_mask(source: Image.Image, kind: str) -> Image.Image:
    hsv = source.convert("HSV")
    width, height = source.size
    mask_values: list[int] = []

    for index, (hue, saturation, value) in enumerate(hsv.getdata()):
        y = index // width
        y_norm = y / max(1, height - 1)

        if kind == "roof":
            selected = saturation > 52 and value > 45 and (hue <= 28 or hue >= 245)
        elif kind == "wall":
            selected = 20 <= hue <= 55 and saturation > 24 and value > 70 and not (saturation > 70 and hue <= 28)
        elif kind == "trim":
            selected = value > 145 and saturation < 80
        elif kind == "window":
            selected = value < 115 and saturation < 105 and 0.15 < y_norm < 0.88
        else:
            selected = value > 28

        mask_values.append(255 if selected else 0)

    mask = Image.new("L", source.size)
    mask.putdata(mask_values)
    return mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(0.65))


def _mask_has_content(mask: Image.Image, min_pixels: int = 256) -> bool:
    histogram = mask.histogram()
    return sum(histogram[64:]) >= min_pixels


def _average_color_with_mask(source: Image.Image, mask: Image.Image, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if not _mask_has_content(mask):
        return fallback

    try:
        mean = ImageStat.Stat(source.convert("RGB"), mask=mask).mean
    except Exception:
        return fallback

    if len(mean) < 3 or sum(mean) <= 1:
        return fallback
    return tuple(max(0, min(255, int(channel))) for channel in mean[:3])


def _material_texture_image(source: Image.Image, kind: str, fallback: tuple[int, int, int]) -> Image.Image:
    mask = _material_mask(source, kind)
    fill = _average_color_with_mask(source, mask, fallback)
    texture = Image.new("RGB", source.size, fill)

    if _mask_has_content(mask):
        texture.paste(source.convert("RGB"), (0, 0), mask)
        texture = ImageEnhance.Contrast(texture).enhance(1.06)
        texture = ImageEnhance.Color(texture).enhance(1.08)
        return texture

    return ImageEnhance.Color(source.convert("RGB")).enhance(1.04)


def _square_seed(image: Image.Image, fallback: tuple[int, int, int], size: int = 1024) -> Image.Image:
    rgb = image.convert("RGB")
    rgb.thumbnail((size, size), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), fallback)
    left = (size - rgb.width) // 2
    top = (size - rgb.height) // 2
    canvas.paste(rgb, (left, top))
    return ImageEnhance.Color(canvas).enhance(1.08)


def _clamp_channel(value: float) -> int:
    return max(0, min(255, int(value)))


def _mix_color(a: tuple[int, int, int], b: tuple[int, int, int], amount: float) -> tuple[int, int, int]:
    return tuple(_clamp_channel(a[index] * (1.0 - amount) + b[index] * amount) for index in range(3))


def _jitter_color(color: tuple[int, int, int], rng: random.Random, amount: int = 22) -> tuple[int, int, int]:
    return tuple(_clamp_channel(channel + rng.randint(-amount, amount)) for channel in color)


def _procedural_material_seed(
    source: Image.Image,
    material: str,
    fallback: tuple[int, int, int],
    size: int = 1024,
) -> Image.Image:
    """
    Build a material-looking img2img seed instead of feeding SD a full object render.

    The seed keeps the source palette, but uses material-specific flat patterns
    so SD is pushed toward usable texture maps rather than another camera view of
    the input image.
    """
    source_rgb = source.convert("RGB")
    mask = _material_mask(source_rgb, material if material in {"roof", "wall", "trim", "window"} else "misc")
    base = _average_color_with_mask(source_rgb, mask, fallback)
    rng = random.Random(f"velocity3d-{material}-{base}")

    image = Image.new("RGB", (size, size), base)
    draw = ImageDraw.Draw(image, "RGBA")

    for y in range(0, size, 4):
        for x in range(0, size, 4):
            shade = rng.randint(-12, 12)
            color = tuple(_clamp_channel(channel + shade) for channel in base)
            draw.rectangle((x, y, x + 3, y + 3), fill=(*color, 68))

    if material == "roof":
        roof_base = _mix_color(base, (190, 68, 28), 0.45)
        image = Image.new("RGB", (size, size), roof_base)
        draw = ImageDraw.Draw(image, "RGBA")
        tile_h = 56
        tile_w = 88
        for row, y in enumerate(range(-tile_h, size + tile_h, tile_h)):
            offset = 0 if row % 2 == 0 else tile_w // 2
            draw.line((0, y + tile_h - 8, size, y + tile_h - 8), fill=(82, 31, 18, 120), width=3)
            for x in range(-tile_w + offset, size + tile_w, tile_w):
                color = _jitter_color(roof_base, rng, 18)
                draw.rounded_rectangle((x, y, x + tile_w, y + tile_h), radius=18, fill=(*color, 180))
                draw.arc((x + 6, y + 8, x + tile_w - 6, y + tile_h + 22), 180, 360, fill=(70, 28, 17, 105), width=3)

    elif material == "wall":
        wall_base = _mix_color(base, (182, 150, 101), 0.28)
        image = Image.new("RGB", (size, size), wall_base)
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(580):
            x = rng.randrange(size)
            y = rng.randrange(size)
            radius = rng.randrange(1, 5)
            color = _jitter_color(wall_base, rng, 26)
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(*color, rng.randrange(28, 92)))

    elif material == "trim":
        trim_base = _mix_color(base, (232, 212, 176), 0.52)
        image = Image.new("RGB", (size, size), trim_base)
        draw = ImageDraw.Draw(image, "RGBA")
        for y in range(0, size, 96):
            draw.rectangle((0, y, size, y + 11), fill=(*_jitter_color(trim_base, rng, 8), 210))
            draw.rectangle((0, y + 54, size, y + 60), fill=(104, 84, 58, 80))
        for x in range(0, size, 128):
            draw.rectangle((x, 0, x + 8, size), fill=(255, 248, 220, 64))

    elif material == "window":
        glass_base = _mix_color(base, (35, 47, 55), 0.66)
        image = Image.new("RGB", (size, size), glass_base)
        draw = ImageDraw.Draw(image, "RGBA")
        for y in range(0, size, 180):
            for x in range(0, size, 180):
                frame = _mix_color(base, (226, 205, 170), 0.48)
                glass = _jitter_color(glass_base, rng, 16)
                draw.rectangle((x + 24, y + 24, x + 156, y + 156), fill=(*glass, 230))
                draw.rectangle((x + 24, y + 24, x + 156, y + 156), outline=(*frame, 230), width=7)
                draw.line((x + 90, y + 24, x + 90, y + 156), fill=(*frame, 210), width=5)
                draw.line((x + 24, y + 90, x + 156, y + 90), fill=(*frame, 210), width=5)

    else:
        misc_base = _mix_color(base, (118, 90, 64), 0.2)
        image = Image.new("RGB", (size, size), misc_base)
        draw = ImageDraw.Draw(image, "RGBA")
        for _ in range(220):
            x0 = rng.randrange(size)
            y0 = rng.randrange(size)
            x1 = x0 + rng.randrange(24, 180)
            y1 = y0 + rng.randrange(3, 18)
            draw.rounded_rectangle((x0, y0, x1, y1), radius=3, fill=(*_jitter_color(misc_base, rng, 28), 70))

    image = image.filter(ImageFilter.GaussianBlur(0.28))
    image = ImageEnhance.Contrast(image).enhance(1.08)
    return ImageEnhance.Color(image).enhance(1.12)


def _blend_generated_with_seed(generated: Image.Image, seed: Image.Image, material: str) -> Image.Image:
    generated = _ensure_colored_albedo(generated, material).resize(seed.size, Image.Resampling.LANCZOS)
    seed = seed.convert("RGB").resize(generated.size, Image.Resampling.LANCZOS)
    blend_amount = {
        "roof": 0.28,
        "wall": 0.34,
        "trim": 0.38,
        "window": 0.42,
        "misc": 0.35,
    }.get(material, 0.35)
    blended = Image.blend(generated.convert("RGB"), seed, blend_amount)
    blended = ImageEnhance.Color(blended).enhance(1.12)
    return ImageEnhance.Contrast(blended).enhance(1.05)


def prepare_material_texture_set(reference_image_bytes: bytes, output_dir: Path, max_size: int = 1536) -> dict[str, Path]:
    """
    Build a material texture folder from a single user reference image.

    The first material keeps the exact projected reference. The derived material
    maps isolate common architectural zones so the exported model is no longer a
    one-texture asset and downstream tools can edit roof/wall/trim/window maps
    independently.
    """
    source = _load_prepared_reference(reference_image_bytes, max_size=max_size)
    output_dir.mkdir(parents=True, exist_ok=True)

    fallback_colors = {
        "facade": _average_opaque_color(source.convert("RGBA")),
        "roof": (171, 67, 28),
        "wall": (190, 134, 55),
        "trim": (224, 203, 166),
        "window": (47, 53, 55),
        "misc": (126, 92, 58),
    }

    textures: dict[str, Path] = {}
    texture_images = {
        "facade": source,
        "roof": _material_texture_image(source, "roof", fallback_colors["roof"]),
        "wall": _material_texture_image(source, "wall", fallback_colors["wall"]),
        "trim": _material_texture_image(source, "trim", fallback_colors["trim"]),
        "window": _material_texture_image(source, "window", fallback_colors["window"]),
        "misc": _material_texture_image(source, "misc", fallback_colors["misc"]),
    }

    for name, image in texture_images.items():
        path = output_dir / f"{name}_albedo.png"
        image.convert("RGB").save(path)
        textures[name] = path

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "type": "velocity3d_material_texture_set",
                "version": 1,
                "textures": {name: path.name for name, path in textures.items()},
                "assignment": {
                    "facade": "front/back projected reference",
                    "roof": "upper/sloped geometry",
                    "wall": "side vertical geometry",
                    "trim": "bounds and cap geometry",
                    "window": "reserved for small facade insert geometry",
                    "misc": "fallback material",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return textures


def generate_material_texture_set(
    reference_image_bytes: bytes,
    output_dir: Path,
    cancellation_event: threading.Event,
    checkpoint: str,
    prompt: str | None = None,
    max_size: int = 1536,
) -> dict[str, Path]:
    """
    Generate editable material textures with Stable Diffusion using the source
    image as img2img guidance.

    The source image is still kept as facade/reference projection, but roof,
    wall, trim, window, and misc material maps are SD-generated from
    reference-derived seeds instead of being direct variants of the input image.
    """
    if cancellation_event.is_set():
        raise ProviderExecutionError("Generation cancelled before material texture synthesis")

    output_dir.mkdir(parents=True, exist_ok=True)
    reference_textures = prepare_material_texture_set(reference_image_bytes, output_dir, max_size=max_size)

    try:
        import torch
    except ImportError as exc:
        raise ProviderDependencyError("Material texture generation requires torch.") from exc

    texture_pipeline = TextureToImagePipeline()
    pipeline = texture_pipeline._get_pipeline(checkpoint, "img2img")
    device = TextureToImagePipeline._device or ("cuda" if torch.cuda.is_available() else "cpu")

    generated_textures: dict[str, Path] = {"facade": reference_textures["facade"]}
    reference_only: dict[str, Path] = {}
    prepared_source = _load_prepared_reference(reference_image_bytes, max_size=max_size)

    for material, reference_path in reference_textures.items():
        reference_copy = output_dir / f"{material}_reference_albedo.png"
        if reference_path.exists() and reference_path != reference_copy:
            Image.open(reference_path).convert("RGB").save(reference_copy)
            reference_only[material] = reference_copy

        if material == "facade":
            continue

        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled during material texture synthesis")

        logger.info("Generating Stable Diffusion %s material texture", material)
        seed_source = Image.open(reference_path).convert("RGB")
        fallback = _average_color_with_mask(seed_source, Image.new("L", seed_source.size, 255), (126, 92, 58))
        seed = _procedural_material_seed(prepared_source, material, fallback)
        seed.save(output_dir / f"{material}_seed.png")
        material_prompt = build_material_texture_prompt(material, prompt)

        generator = None
        if device == "cuda":
            seed_value = 137 + sum(ord(ch) for ch in material)
            generator = torch.Generator(device).manual_seed(seed_value)

        try:
            result = pipeline(
                prompt=material_prompt,
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                image=seed,
                strength=0.72 if material in {"roof", "wall", "window"} else 0.64,
                guidance_scale=7.8,
                num_inference_steps=30,
                generator=generator,
            )
        except Exception as exc:
            raise ProviderExecutionError(f"Stable Diffusion material texture generation failed for {material}: {exc}") from exc

        image = result.images[0]
        if not isinstance(image, Image.Image):
            raise ProviderExecutionError(f"Stable Diffusion did not return an image for {material}")

        generated = _blend_generated_with_seed(image, seed, material)
        output_path = output_dir / f"{material}_albedo.png"
        generated.save(output_path)
        generated_textures[material] = output_path
        logger.info("Stable Diffusion %s material texture written to %s", material, output_path)

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "type": "velocity3d_sd_material_texture_set",
                "version": 2,
                "checkpoint": checkpoint,
                "textures": {name: path.name for name, path in generated_textures.items()},
                "reference_textures": {name: path.name for name, path in reference_only.items()},
                "source": "stable_diffusion_img2img_from_reference_material_seeds",
                "assignment": {
                    "facade": "source reference projection",
                    "roof": "SD-generated roof material from source-guided seed",
                    "wall": "SD-generated wall material from source-guided seed",
                    "trim": "SD-generated trim material from source-guided seed",
                    "window": "SD-generated window material from source-guided seed",
                    "misc": "SD-generated fallback material from source-guided seed",
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    return generated_textures


def checkpoint_dir_for(model_id: str) -> Path:
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return local_path

    safe_name = "__".join(safe_stem(part, "checkpoint") for part in model_id.replace("\\", "/").split("/"))
    return checkpoints_root() / safe_name


def ensure_checkpoint_local(model_id: str) -> Path:
    local_path = Path(model_id).expanduser()
    if local_path.exists():
        return local_path

    destination = checkpoint_dir_for(model_id)
    model_index = destination / "model_index.json"
    if model_index.exists():
        return destination

    destination.mkdir(parents=True, exist_ok=True)
    configure_huggingface_cache()
    logger.info("Downloading texture checkpoint %s to %s", model_id, destination)

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - depends on local runtime
        raise ProviderDependencyError(
            "Texture checkpoint download requires huggingface-hub. "
            "Install the texture dependencies before enabling AI textures."
        ) from exc

    try:
        snapshot_download(
            repo_id=model_id,
            local_dir=str(destination),
        )
    except Exception as exc:  # pragma: no cover - depends on network/hub
        raise ProviderExecutionError(f"Could not download texture checkpoint {model_id}: {exc}") from exc

    if not model_index.exists():
        raise ProviderExecutionError(
            f"Texture checkpoint {model_id} downloaded to {destination}, but model_index.json was not found."
        )

    logger.info("Texture checkpoint ready at %s", destination)
    return destination


class TextureToImagePipeline:
    _pipeline = None
    _checkpoint: str | None = None
    _pipeline_mode: str | None = None
    _device: str | None = None
    _lock = threading.Lock()

    def _load_pipeline(self, checkpoint: str, mode: str):
        try:
            import torch
            from diffusers import AutoPipelineForImage2Image, AutoPipelineForText2Image
        except ImportError as exc:  # pragma: no cover - depends on local runtime
            raise ProviderDependencyError(
                "Texture generation requires diffusers and safetensors. "
                "Install them in the backend environment before enabling AI textures."
            ) from exc

        device = "cuda" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if device == "cuda" else torch.float32
        checkpoint_path = ensure_checkpoint_local(checkpoint)

        pipeline_cls = AutoPipelineForImage2Image if mode == "img2img" else AutoPipelineForText2Image
        pipeline = pipeline_cls.from_pretrained(
            str(checkpoint_path),
            torch_dtype=torch_dtype,
            use_safetensors=True,
            local_files_only=True,
        )
        pipeline.set_progress_bar_config(disable=True)

        if device == "cuda":
            try:
                pipeline.enable_model_cpu_offload()
            except Exception:
                pipeline.to(device)
        else:
            pipeline.to(device)

        TextureToImagePipeline._device = device
        return pipeline

    def _get_pipeline(self, checkpoint: str, mode: str):
        cls = type(self)
        if cls._pipeline is not None and cls._checkpoint == checkpoint and cls._pipeline_mode == mode:
            return cls._pipeline

        with cls._lock:
            if cls._pipeline is not None and cls._checkpoint == checkpoint and cls._pipeline_mode == mode:
                return cls._pipeline

            cls._pipeline = self._load_pipeline(checkpoint, mode)
            cls._checkpoint = checkpoint
            cls._pipeline_mode = mode
            return cls._pipeline

    def prepare(self, checkpoint: str | None = None, mode: str = "text2img") -> Path:
        model_id = checkpoint or DEFAULT_TEXTURE_CHECKPOINT
        checkpoint_path = ensure_checkpoint_local(model_id)
        self._get_pipeline(model_id, mode)
        return checkpoint_path

    def release(self) -> None:
        cls = type(self)
        with cls._lock:
            pipeline = cls._pipeline
            cls._pipeline = None
            cls._checkpoint = None
            cls._pipeline_mode = None
            cls._device = None

        if pipeline is not None:
            del pipeline

        try:
            import gc
            import torch

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            logger.debug("Could not fully release texture pipeline memory", exc_info=True)

    def generate(
        self,
        prompt: str | None,
        output_path: Path,
        cancellation_event: threading.Event,
        checkpoint: str | None = None,
        uv_layout_path: Path | None = None,
    ) -> Path:
        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled before texture synthesis")

        model_id = checkpoint or DEFAULT_TEXTURE_CHECKPOINT
        use_uv_guide = uv_layout_path is not None and uv_layout_path.exists()
        pipeline = self._get_pipeline(model_id, "text2img")
        texture_prompt = build_texture_prompt(prompt, uv_guided=use_uv_guide)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            import torch

            generator = None
            device = type(self)._device or ("cuda" if torch.cuda.is_available() else "cpu")
            if device == "cuda":
                generator = torch.Generator(device).manual_seed(17)

            common_kwargs = {
                "prompt": texture_prompt,
                "negative_prompt": DEFAULT_NEGATIVE_PROMPT,
                "num_inference_steps": 32,
                "guidance_scale": 8.5,
                "generator": generator,
            }

            result = pipeline(
                **common_kwargs,
                height=1024,
                width=1024,
            )
        except Exception as exc:  # pragma: no cover - depends on local runtime/model
            raise ProviderExecutionError(f"Texture generation failed: {exc}") from exc

        if cancellation_event.is_set():
            raise ProviderExecutionError("Generation cancelled after texture synthesis")

        image = result.images[0]
        if not isinstance(image, Image.Image):
            raise ProviderExecutionError("Texture generation did not return an image")

        image = _apply_uv_mask(image, uv_layout_path, prompt)
        image.save(output_path)
        return output_path
