"""
bpy_worker.py — runs as a standalone subprocess, receives a task via stdin JSON,
executes bpy operations, and writes the result to stdout JSON.

This isolates bpy completely from the uvicorn event loop.

Protocol:
  stdin:  single JSON line with task description
  stdout: single JSON line with {"ok": true, "output_path": "..."} or {"ok": false, "error": "..."}
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def _validate_glb(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) == b"glTF"
    except OSError:
        return False


def task_process(task: dict) -> str:
    """Import vertices/faces, clean up, export to GLB. Returns output path."""
    import bpy
    import numpy as np

    vertices = task["vertices"]   # list of [x, y, z]
    faces = task["faces"]         # list of [a, b, c]
    output_path = task["output_path"]
    name = task.get("name", "mesh")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)

    mesh = bpy.data.meshes.new(name)
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    mesh.from_pydata(
        [tuple(v) for v in vertices],
        [],
        [tuple(f) for f in faces],
    )
    mesh.update()

    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=0.0001)
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        use_selection=False,
    )

    if not _validate_glb(output_path):
        raise RuntimeError(f"Output is not a valid GLB: {output_path}")

    return output_path


def task_export(task: dict) -> str:
    """Re-export an existing GLB to glb/obj/fbx. Returns output path."""
    import bpy

    source_glb = task["source_glb"]
    output_path = task["output_path"]
    fmt = task["format"]  # "glb" | "obj" | "fbx"

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source_glb)

    if fmt == "glb":
        bpy.ops.export_scene.gltf(filepath=output_path, export_format="GLB")
    elif fmt == "obj":
        bpy.ops.wm.obj_export(filepath=output_path)
    elif fmt == "fbx":
        bpy.ops.export_scene.fbx(filepath=output_path)
    else:
        raise ValueError(f"Unsupported format: {fmt}")

    return output_path


def _ensure_uvs(obj) -> None:
    import bpy

    if obj.type != "MESH":
        return

    if obj.data.uv_layers:
        return

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.02)
    bpy.ops.object.mode_set(mode="OBJECT")


def _create_velocity_uv_atlas(obj) -> None:
    import bpy

    if obj.type != "MESH":
        return

    mesh = obj.data
    uv_layer = mesh.uv_layers.get("Velocity3D UV Atlas")
    if uv_layer is None:
        uv_layer = mesh.uv_layers.new(name="Velocity3D UV Atlas")
    mesh.uv_layers.active = uv_layer

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.uv.smart_project(angle_limit=66.0, island_margin=0.035)
    bpy.ops.object.mode_set(mode="OBJECT")


def _draw_uv_layout(output_path: str, size: int = 1024) -> None:
    from PIL import Image, ImageDraw

    import bpy

    image = Image.new("RGB", (size, size), "black")
    draw = ImageDraw.Draw(image, "RGBA")

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        mesh = obj.data
        if not mesh.uv_layers:
            continue

        uv_layer = mesh.uv_layers.active.data
        for polygon in mesh.polygons:
            points = []
            for loop_index in polygon.loop_indices:
                uv = uv_layer[loop_index].uv
                x = max(0, min(size - 1, int(uv.x * (size - 1))))
                y = max(0, min(size - 1, int((1.0 - uv.y) * (size - 1))))
                points.append((x, y))

            if len(points) >= 3:
                draw.polygon(points, fill=(238, 238, 238, 255))
                draw.line(points + [points[0]], fill=(255, 255, 255, 255), width=2)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def task_prepare_texture_target(task: dict) -> str:
    import bpy

    source_glb = task["source_glb"]
    output_path = task["output_path"]
    uv_layout_path = task["uv_layout_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(uv_layout_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source_glb)

    for obj in bpy.context.scene.objects:
        if obj.type == "MESH":
            _ensure_uvs(obj)

    _draw_uv_layout(uv_layout_path, size=1024)

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )

    if not _validate_glb(output_path):
        raise RuntimeError(f"UV-prepared output is not a valid GLB: {output_path}")

    return output_path


def _apply_texture_to_material(material, image) -> None:
    import bpy

    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")

    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")

    tex_node = next((node for node in nodes if node.type == "TEX_IMAGE" and node.name == "Velocity3D Texture"), None)
    if tex_node is None:
        tex_node = nodes.new("ShaderNodeTexImage")
        tex_node.name = "Velocity3D Texture"

    tex_node.image = image
    tex_node.interpolation = "Smart"
    tex_node.extension = "CLIP"

    for link in list(principled.inputs["Base Color"].links):
        links.remove(link)
    links.new(tex_node.outputs["Color"], principled.inputs["Base Color"])

    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])

    principled.inputs["Roughness"].default_value = 0.62
    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = 0.0
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.35


def _principled_node(material):
    material.use_nodes = True
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    output = next((node for node in nodes if node.type == "OUTPUT_MATERIAL"), None)
    if output is None:
        output = nodes.new("ShaderNodeOutputMaterial")

    principled = next((node for node in nodes if node.type == "BSDF_PRINCIPLED"), None)
    if principled is None:
        principled = nodes.new("ShaderNodeBsdfPrincipled")

    for link in list(output.inputs["Surface"].links):
        links.remove(link)
    links.new(principled.outputs["BSDF"], output.inputs["Surface"])
    return principled


def _apply_pbr_maps_to_material(material, albedo_image, roughness_image=None, normal_image=None) -> None:
    import bpy

    material.diffuse_color = (1.0, 1.0, 1.0, 1.0)
    principled = _principled_node(material)
    nodes = material.node_tree.nodes
    links = material.node_tree.links

    albedo_node = nodes.new("ShaderNodeTexImage")
    albedo_node.name = "Velocity3D UV Albedo"
    albedo_node.image = albedo_image
    albedo_node.interpolation = "Smart"
    albedo_node.extension = "CLIP"
    for link in list(principled.inputs["Base Color"].links):
        links.remove(link)
    links.new(albedo_node.outputs["Color"], principled.inputs["Base Color"])

    if "Metallic" in principled.inputs:
        principled.inputs["Metallic"].default_value = 0.0
    if "Roughness" in principled.inputs:
        principled.inputs["Roughness"].default_value = 0.68
    if "Specular IOR Level" in principled.inputs:
        principled.inputs["Specular IOR Level"].default_value = 0.42

    if roughness_image is not None and "Roughness" in principled.inputs:
        roughness_node = nodes.new("ShaderNodeTexImage")
        roughness_node.name = "Velocity3D UV Roughness"
        roughness_node.image = roughness_image
        roughness_node.interpolation = "Smart"
        roughness_node.extension = "CLIP"
        roughness_image.colorspace_settings.name = "Non-Color"
        for link in list(principled.inputs["Roughness"].links):
            links.remove(link)
        links.new(roughness_node.outputs["Color"], principled.inputs["Roughness"])

    if normal_image is not None and "Normal" in principled.inputs:
        normal_node = nodes.new("ShaderNodeTexImage")
        normal_node.name = "Velocity3D UV Normal"
        normal_node.image = normal_image
        normal_node.interpolation = "Smart"
        normal_node.extension = "CLIP"
        normal_image.colorspace_settings.name = "Non-Color"

        normal_map = nodes.new("ShaderNodeNormalMap")
        normal_map.inputs["Strength"].default_value = 0.32
        links.new(normal_node.outputs["Color"], normal_map.inputs["Color"])
        for link in list(principled.inputs["Normal"].links):
            links.remove(link)
        links.new(normal_map.outputs["Normal"], principled.inputs["Normal"])


def task_apply_texture(task: dict) -> str:
    import bpy

    source_glb = task["source_glb"]
    texture_image = task["texture_image"]
    output_path = task["output_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source_glb)

    image = bpy.data.images.load(texture_image)
    image.colorspace_settings.name = "sRGB"
    try:
        image.pack()
    except Exception:
        pass

    for obj in bpy.context.scene.objects:
        if obj.type != "MESH":
            continue

        _ensure_uvs(obj)

        if not obj.data.materials:
            material = bpy.data.materials.new(name=f"{obj.name}_Material")
            obj.data.materials.append(material)

        for slot_index, material in enumerate(obj.data.materials):
            if material is None:
                material = bpy.data.materials.new(name=f"{obj.name}_Material")
                obj.data.materials[slot_index] = material
            _apply_texture_to_material(material, image)

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )

    if not _validate_glb(output_path):
        raise RuntimeError(f"Output is not a valid GLB: {output_path}")

    return output_path


def _mesh_world_bounds(objects) -> tuple[float, float, float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    zs: list[float] = []

    for obj in objects:
        for vertex in obj.data.vertices:
            world = obj.matrix_world @ vertex.co
            xs.append(float(world.x))
            ys.append(float(world.y))
            zs.append(float(world.z))

    if not xs:
        raise RuntimeError("No mesh vertices were available for reference projection")

    return min(xs), max(xs), min(ys), max(ys), min(zs), max(zs)


def _set_reference_projection_uvs(objects) -> None:
    bounds = _mesh_world_bounds(objects)
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)
    range_z = max(max_z - min_z, 1e-6)
    use_y_for_u = range_y > range_x * 1.2

    for obj in objects:
        mesh = obj.data
        uv_layer = mesh.uv_layers.get("Velocity3D Reference Projection")
        if uv_layer is None:
            uv_layer = mesh.uv_layers.new(name="Velocity3D Reference Projection")

        mesh.uv_layers.active = uv_layer
        for polygon in mesh.polygons:
            for loop_index in polygon.loop_indices:
                loop = mesh.loops[loop_index]
                world = obj.matrix_world @ mesh.vertices[loop.vertex_index].co
                if use_y_for_u:
                    u = (float(world.y) - min_y) / range_y
                else:
                    u = (float(world.x) - min_x) / range_x
                v = (float(world.z) - min_z) / range_z
                uv_layer.data[loop_index].uv = (
                    max(0.0, min(1.0, u)),
                    max(0.0, min(1.0, v)),
                )


def task_project_reference_texture(task: dict) -> str:
    import bpy

    source_glb = task["source_glb"]
    reference_image = task["reference_image"]
    output_path = task["output_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source_glb)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Imported GLB did not contain mesh objects")

    image = bpy.data.images.load(reference_image)
    image.colorspace_settings.name = "sRGB"
    try:
        image.pack()
    except Exception:
        pass

    _set_reference_projection_uvs(mesh_objects)

    material = bpy.data.materials.new(name="Velocity3D Reference Texture")
    _apply_texture_to_material(material, image)

    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )

    if not _validate_glb(output_path):
        raise RuntimeError(f"Reference textured output is not a valid GLB: {output_path}")

    return output_path


def task_project_reference_texture_baked(task: dict) -> str:
    import bpy

    source_glb = task["source_glb"]
    albedo_image_path = task["albedo_image"]
    roughness_image_path = task.get("roughness_image")
    output_path = task["output_path"]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source_glb)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Imported GLB did not contain mesh objects")

    _set_reference_projection_uvs(mesh_objects)

    albedo = bpy.data.images.load(albedo_image_path)
    albedo.colorspace_settings.name = "sRGB"
    roughness = None
    if roughness_image_path:
        roughness = bpy.data.images.load(roughness_image_path)
        roughness.colorspace_settings.name = "Non-Color"

    for image in (albedo, roughness):
        if image is None:
            continue
        try:
            image.pack()
        except Exception:
            pass

    material = bpy.data.materials.new(name="Velocity3D Projected Reference")
    _apply_pbr_maps_to_material(material, albedo, roughness, None)

    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.material_index = 0

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )

    if not _validate_glb(output_path):
        raise RuntimeError(f"Projected textured output is not a valid GLB: {output_path}")

    return output_path


def _projection_uses_y_for_u(bounds: tuple[float, float, float, float, float, float]) -> bool:
    min_x, max_x, min_y, max_y, _min_z, _max_z = bounds
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)
    return range_y > range_x * 1.2


def _polygon_world_center_and_normal(obj, polygon):
    center = None
    for loop_index in polygon.loop_indices:
        vertex = obj.data.vertices[obj.data.loops[loop_index].vertex_index]
        world = obj.matrix_world @ vertex.co
        center = world.copy() if center is None else center + world
    center = center / max(1, len(polygon.loop_indices))

    normal = obj.matrix_world.to_3x3() @ polygon.normal
    normal.normalize()
    return center, normal


def _classify_material_zone(obj, polygon, bounds: tuple[float, float, float, float, float, float]) -> str:
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)
    range_z = max(max_z - min_z, 1e-6)
    use_y_for_u = _projection_uses_y_for_u(bounds)

    center, normal = _polygon_world_center_and_normal(obj, polygon)
    x_norm = (float(center.x) - min_x) / range_x
    y_norm = (float(center.y) - min_y) / range_y
    z_norm = (float(center.z) - min_z) / range_z

    depth_normal = abs(float(normal.x)) if use_y_for_u else abs(float(normal.y))
    length_normal = abs(float(normal.y)) if use_y_for_u else abs(float(normal.x))
    vertical = abs(float(normal.z)) < 0.58
    front_back = vertical and depth_normal >= 0.42
    side_wall = vertical and length_normal >= 0.42
    near_horizontal_edge = x_norm < 0.055 or x_norm > 0.945 or y_norm < 0.055 or y_norm > 0.945
    near_vertical_cap = z_norm < 0.065 or z_norm > 0.91

    if z_norm > 0.64 and (float(normal.z) > 0.12 or not front_back):
        return "roof"

    if not front_back and (near_vertical_cap or (near_horizontal_edge and z_norm > 0.12)):
        return "trim"

    if front_back:
        return "facade"

    if side_wall:
        if near_vertical_cap or near_horizontal_edge:
            return "trim"
        return "wall"

    if z_norm > 0.82:
        return "roof"

    return "misc"


def _polygon_projection_uv(obj, polygon):
    mesh = obj.data
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    if uv_layer is None:
        return None

    u_total = 0.0
    v_total = 0.0
    count = 0
    for loop_index in polygon.loop_indices:
        uv = uv_layer[loop_index].uv
        u_total += float(uv.x)
        v_total += float(uv.y)
        count += 1

    if count == 0:
        return None
    return (u_total / count, v_total / count)


def _looks_like_window_pixel(pixel: tuple[int, int, int]) -> bool:
    red, green, blue = pixel[:3]
    value = max(red, green, blue)
    chroma = value - min(red, green, blue)
    return value < 122 and chroma < 58


def _facade_uv_is_window(facade_image, uv) -> bool:
    if facade_image is None or uv is None:
        return False

    u, v = uv
    width, height = facade_image.size
    x = max(0, min(width - 1, int(u * (width - 1))))
    y = max(0, min(height - 1, int((1.0 - v) * (height - 1))))
    return _looks_like_window_pixel(facade_image.getpixel((x, y)))


def _load_pil_texture(path: str):
    from PIL import Image, ImageEnhance

    image = Image.open(path).convert("RGB")
    image = ImageEnhance.Color(image).enhance(1.08)
    return image


def _average_pil_color(image) -> tuple[int, int, int]:
    from PIL import ImageStat

    stat = ImageStat.Stat(image.convert("RGB"))
    return tuple(max(0, min(255, int(channel))) for channel in stat.mean[:3])


def _uv_points_for_polygon(obj, polygon, atlas_size: int) -> list[tuple[int, int]]:
    mesh = obj.data
    uv_layer = mesh.uv_layers.active.data if mesh.uv_layers.active else None
    if uv_layer is None:
        return []

    points: list[tuple[int, int]] = []
    for loop_index in polygon.loop_indices:
        uv = uv_layer[loop_index].uv
        u = max(0.0, min(1.0, float(uv.x)))
        v = max(0.0, min(1.0, float(uv.y)))
        points.append((
            max(0, min(atlas_size - 1, int(u * (atlas_size - 1)))),
            max(0, min(atlas_size - 1, int((1.0 - v) * (atlas_size - 1)))),
        ))
    return points


def _make_roughness_from_albedo(albedo):
    from PIL import ImageEnhance, ImageOps

    gray = ImageOps.grayscale(albedo)
    gray = ImageOps.autocontrast(gray)
    roughness = gray.point(lambda value: max(96, min(238, int(202 - (value - 128) * 0.18))), "L")
    return ImageEnhance.Contrast(roughness).enhance(0.72)


def _make_normal_from_albedo(albedo, strength: float = 1.8):
    from PIL import Image, ImageFilter, ImageOps
    import math

    height = ImageOps.grayscale(albedo).filter(ImageFilter.GaussianBlur(0.7))
    width, height_px = height.size
    src = height.load()
    normal = Image.new("RGB", (width, height_px), (128, 128, 255))
    dst = normal.load()

    for y in range(height_px):
        y0 = max(0, y - 1)
        y1 = min(height_px - 1, y + 1)
        for x in range(width):
            x0 = max(0, x - 1)
            x1 = min(width - 1, x + 1)
            dx = (src[x1, y] - src[x0, y]) / 255.0
            dy = (src[x, y1] - src[x, y0]) / 255.0
            nx = -dx * strength
            ny = -dy * strength
            nz = 1.0
            length = max(1e-6, math.sqrt(nx * nx + ny * ny + nz * nz))
            dst[x, y] = (
                int((nx / length * 0.5 + 0.5) * 255),
                int((ny / length * 0.5 + 0.5) * 255),
                int((nz / length * 0.5 + 0.5) * 255),
            )

    return normal


def _bake_material_uv_atlas(mesh_objects, zone_by_polygon, texture_paths: dict, output_path: str, atlas_size: int = 1024):
    from PIL import Image, ImageDraw, ImageFilter

    texture_images = {}
    for zone, path in texture_paths.items():
        if not path:
            continue
        try:
            texture_images[zone] = _load_pil_texture(path)
        except Exception:
            continue

    if not texture_images:
        raise RuntimeError("No readable material textures were available for UV baking")

    fallback = texture_images.get("misc") or texture_images.get("facade") or next(iter(texture_images.values()))
    atlas = Image.new("RGB", (atlas_size, atlas_size), _average_pil_color(fallback))

    for obj in mesh_objects:
        mesh = obj.data
        if not mesh.uv_layers.active:
            continue

        for polygon in mesh.polygons:
            points = _uv_points_for_polygon(obj, polygon, atlas_size)
            if len(points) < 3:
                continue

            zone = zone_by_polygon.get(obj.name, {}).get(polygon.index, "misc")
            texture = texture_images.get(zone) or texture_images.get("misc") or texture_images.get("facade") or fallback

            min_x = max(0, min(point[0] for point in points))
            max_x = min(atlas_size - 1, max(point[0] for point in points))
            min_y = max(0, min(point[1] for point in points))
            max_y = min(atlas_size - 1, max(point[1] for point in points))
            box_width = max(1, max_x - min_x + 1)
            box_height = max(1, max_y - min_y + 1)

            patch = texture.resize((box_width, box_height), Image.Resampling.LANCZOS)
            mask = Image.new("L", (box_width, box_height), 0)
            shifted = [(x - min_x, y - min_y) for x, y in points]
            draw = ImageDraw.Draw(mask)
            draw.polygon(shifted, fill=255)
            draw.line(shifted + [shifted[0]], fill=255, width=3)
            atlas.paste(patch, (min_x, min_y), mask)

    atlas = atlas.filter(ImageFilter.UnsharpMask(radius=1.0, percent=80, threshold=3))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output_path)
    return atlas


def _material_tile_rects() -> dict[str, tuple[float, float, float, float]]:
    return {
        "facade": (0.000, 0.500, 0.333, 1.000),
        "roof": (0.333, 0.500, 0.666, 1.000),
        "wall": (0.666, 0.500, 1.000, 1.000),
        "trim": (0.000, 0.000, 0.333, 0.500),
        "window": (0.333, 0.000, 0.666, 0.500),
        "misc": (0.666, 0.000, 1.000, 0.500),
    }


def _paste_texture_tile(atlas, texture, rect: tuple[float, float, float, float], margin: int) -> None:
    from PIL import Image

    atlas_size = atlas.size[0]
    u0, v0, u1, v1 = rect
    left = int(u0 * atlas_size) + margin
    right = int(u1 * atlas_size) - margin
    top = int((1.0 - v1) * atlas_size) + margin
    bottom = int((1.0 - v0) * atlas_size) - margin
    width = max(1, right - left)
    height = max(1, bottom - top)
    atlas.paste(texture.resize((width, height), Image.Resampling.LANCZOS), (left, top))


def _bake_material_tile_atlas(texture_paths: dict, output_path: str, atlas_size: int = 1024):
    from PIL import Image, ImageDraw, ImageFilter

    texture_images = {}
    for zone, path in texture_paths.items():
        if not path:
            continue
        try:
            texture_images[zone] = _load_pil_texture(path)
        except Exception:
            continue

    if not texture_images:
        raise RuntimeError("No readable material textures were available for UV atlas baking")

    fallback = texture_images.get("misc") or texture_images.get("facade") or next(iter(texture_images.values()))
    atlas = Image.new("RGB", (atlas_size, atlas_size), _average_pil_color(fallback))
    draw = ImageDraw.Draw(atlas, "RGBA")
    margin = max(8, atlas_size // 96)

    for zone, rect in _material_tile_rects().items():
        texture = texture_images.get(zone) or texture_images.get("misc") or texture_images.get("facade") or fallback
        _paste_texture_tile(atlas, texture, rect, margin)
        u0, v0, u1, v1 = rect
        left = int(u0 * atlas_size)
        right = int(u1 * atlas_size)
        top = int((1.0 - v1) * atlas_size)
        bottom = int((1.0 - v0) * atlas_size)
        draw.rectangle((left, top, right, bottom), outline=(255, 255, 255, 42), width=2)

    atlas = atlas.filter(ImageFilter.UnsharpMask(radius=1.0, percent=70, threshold=3))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    atlas.save(output_path)
    return atlas


def _set_material_tile_uvs(mesh_objects, zone_by_polygon, bounds: tuple[float, float, float, float, float, float]) -> None:
    min_x, max_x, min_y, max_y, min_z, max_z = bounds
    range_x = max(max_x - min_x, 1e-6)
    range_y = max(max_y - min_y, 1e-6)
    range_z = max(max_z - min_z, 1e-6)
    tile_rects = _material_tile_rects()
    padding = 0.022

    def frac(value: float) -> float:
        return value - int(value)

    for obj in mesh_objects:
        mesh = obj.data
        uv_layer = mesh.uv_layers.get("Velocity3D Material Atlas")
        if uv_layer is None:
            uv_layer = mesh.uv_layers.new(name="Velocity3D Material Atlas")
        mesh.uv_layers.active = uv_layer

        for polygon in mesh.polygons:
            zone = zone_by_polygon.get(obj.name, {}).get(polygon.index, "misc")
            rect = tile_rects.get(zone, tile_rects["misc"])
            u0, v0, u1, v1 = rect
            width = max(1e-6, u1 - u0)
            height = max(1e-6, v1 - v0)

            center, normal = _polygon_world_center_and_normal(obj, polygon)
            use_y_for_u = _projection_uses_y_for_u(bounds)
            for loop_index in polygon.loop_indices:
                vertex = mesh.vertices[mesh.loops[loop_index].vertex_index]
                world = obj.matrix_world @ vertex.co

                if zone == "roof":
                    local_u = (float(world.x) - min_x) / range_x * 3.5
                    local_v = (float(world.y) - min_y) / range_y * 3.5
                elif zone in {"facade", "window"}:
                    local_u = ((float(world.y) - min_y) / range_y if use_y_for_u else (float(world.x) - min_x) / range_x) * 1.2
                    local_v = (float(world.z) - min_z) / range_z * 1.15
                elif zone == "wall":
                    local_u = ((float(world.x) - min_x) / range_x if use_y_for_u else (float(world.y) - min_y) / range_y) * 1.8
                    local_v = (float(world.z) - min_z) / range_z * 1.25
                elif zone == "trim":
                    axis = abs(float(normal.x)) > abs(float(normal.y))
                    local_u = ((float(world.y) - min_y) / range_y if axis else (float(world.x) - min_x) / range_x) * 2.2
                    local_v = (float(world.z) - min_z) / range_z * 1.8
                else:
                    local_u = (float(world.x) - min_x) / range_x * 2.0
                    local_v = (float(world.y) - min_y) / range_y * 2.0

                u = u0 + padding * width + frac(local_u) * width * (1.0 - padding * 2.0)
                v = v0 + padding * height + frac(local_v) * height * (1.0 - padding * 2.0)
                uv_layer.data[loop_index].uv = (max(0.0, min(1.0, u)), max(0.0, min(1.0, v)))


def task_project_material_textures(task: dict) -> str:
    import bpy
    from PIL import Image

    source_glb = task["source_glb"]
    texture_paths = task["texture_paths"]
    output_path = task["output_path"]
    uv_albedo_path = task.get("uv_albedo_path") or str(Path(output_path).with_suffix(".uv_albedo.png"))
    uv_roughness_path = task.get("uv_roughness_path") or str(Path(output_path).with_suffix(".uv_roughness.png"))
    uv_normal_path = task.get("uv_normal_path") or str(Path(output_path).with_suffix(".uv_normal.png"))

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.import_scene.gltf(filepath=source_glb)

    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]
    if not mesh_objects:
        raise RuntimeError("Imported GLB did not contain mesh objects")

    _set_reference_projection_uvs(mesh_objects)
    bounds = _mesh_world_bounds(mesh_objects)
    facade_sampler = None
    if texture_paths.get("facade"):
        try:
            facade_sampler = Image.open(texture_paths["facade"]).convert("RGB")
        except Exception:
            facade_sampler = None

    zone_by_polygon: dict[str, dict[int, str]] = {}
    for obj in mesh_objects:
        zone_by_polygon[obj.name] = {}
        for polygon in obj.data.polygons:
            zone = _classify_material_zone(obj, polygon, bounds)
            if zone == "facade" and _facade_uv_is_window(facade_sampler, _polygon_projection_uv(obj, polygon)):
                zone = "window"
            zone_by_polygon[obj.name][polygon.index] = zone

    _set_material_tile_uvs(mesh_objects, zone_by_polygon, bounds)
    uv_albedo = _bake_material_tile_atlas(texture_paths, uv_albedo_path)
    roughness = _make_roughness_from_albedo(uv_albedo)
    normal = _make_normal_from_albedo(uv_albedo)
    Path(uv_roughness_path).parent.mkdir(parents=True, exist_ok=True)
    roughness.save(uv_roughness_path)
    normal.save(uv_normal_path)

    albedo_image = bpy.data.images.load(uv_albedo_path)
    albedo_image.colorspace_settings.name = "sRGB"
    roughness_image = bpy.data.images.load(uv_roughness_path)
    roughness_image.colorspace_settings.name = "Non-Color"
    normal_image = bpy.data.images.load(uv_normal_path)
    normal_image.colorspace_settings.name = "Non-Color"
    for image in (albedo_image, roughness_image, normal_image):
        try:
            image.pack()
        except Exception:
            pass

    material = bpy.data.materials.new(name="Velocity3D UV PBR Material")
    _apply_pbr_maps_to_material(material, albedo_image, roughness_image, normal_image)

    for obj in mesh_objects:
        obj.data.materials.clear()
        obj.data.materials.append(material)
        for polygon in obj.data.polygons:
            polygon.material_index = 0

    bpy.ops.export_scene.gltf(
        filepath=output_path,
        export_format="GLB",
        export_image_format="AUTO",
        export_texcoords=True,
        export_normals=True,
        export_materials="EXPORT",
    )

    if not _validate_glb(output_path):
        raise RuntimeError(f"Material textured output is not a valid GLB: {output_path}")

    return output_path


def _redirect_stdout_to_stderr() -> int | None:
    """
    Keep the worker stdout channel reserved for protocol JSON.

    Blender's import/export operators can write directly to the process stdout
    file descriptor, bypassing Python's sys.stdout object. Redirecting fd 1 to
    stderr keeps those logs visible to the parent while preserving a duplicate
    of the original stdout pipe for the final JSON response.
    """
    try:
        sys.stdout.flush()
        original_stdout_fd = os.dup(sys.stdout.fileno())
        os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
        return original_stdout_fd
    except (OSError, ValueError):
        return None


def _emit_protocol_json(stdout_fd: int | None, payload: dict) -> None:
    line = (json.dumps(payload) + "\n").encode("utf-8")
    if stdout_fd is None:
        print(line.decode("utf-8"), end="", flush=True)
        return

    os.write(stdout_fd, line)


def main() -> None:
    raw = sys.stdin.readline()
    protocol_stdout_fd = _redirect_stdout_to_stderr()
    try:
        task = json.loads(raw)
        op = task.get("op")

        if op == "process":
            output_path = task_process(task)
        elif op == "export":
            output_path = task_export(task)
        elif op == "prepare_texture_target":
            output_path = task_prepare_texture_target(task)
        elif op == "apply_texture":
            output_path = task_apply_texture(task)
        elif op == "project_reference_texture":
            output_path = task_project_reference_texture(task)
        elif op == "project_reference_texture_baked":
            output_path = task_project_reference_texture_baked(task)
        elif op == "project_material_textures":
            output_path = task_project_material_textures(task)
        else:
            raise ValueError(f"Unknown op: {op}")

        _emit_protocol_json(protocol_stdout_fd, {"ok": True, "output_path": output_path})

    except Exception as exc:
        _emit_protocol_json(
            protocol_stdout_fd,
            {"ok": False, "error": str(exc), "traceback": traceback.format_exc()},
        )
        sys.exit(1)
    finally:
        if protocol_stdout_fd is not None:
            try:
                os.close(protocol_stdout_fd)
            except OSError:
                pass


if __name__ == "__main__":
    main()
