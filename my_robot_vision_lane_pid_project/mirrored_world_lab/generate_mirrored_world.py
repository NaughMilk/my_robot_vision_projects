#!/usr/bin/env python3
from __future__ import annotations

import argparse
import heapq
import json
import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


MODEL_SIZE_M = 6.0
MAP_TEXTURE_SIZE = 256.0
DEFAULT_MAP_POSE = (0.0, 0.3, 0.0, 0.0, 0.0, 0.0)
DEFAULT_CLEARANCE_PX = 12
CURB_VISUAL_BUFFER_PX = 3
ROUTE_CENTERLINE_WEIGHT = 18.0
STRAIGHT_ROUTE_AXIS_TOLERANCE_PX = 1.0
STRAIGHT_ROUTE_SIMPLIFY_MAX_LATERAL_PX = 4.0
SIGN_MODEL_PREFIXES = ("sign_",)
TRAFFIC_MODEL_PREFIXES = ("traffic_light_",)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def find_child(element: ET.Element, child_name: str) -> ET.Element | None:
    for child in element:
        if local_name(child.tag) == child_name:
            return child
    return None


def child_text(element: ET.Element, child_name: str) -> str | None:
    child = find_child(element, child_name)
    return None if child is None else child.text


def parse_pose(text: str | None) -> list[float]:
    values = [float(value) for value in (text or "0 0 0 0 0 0").split()]
    values.extend([0.0] * (6 - len(values)))
    return values[:6]


def pose_to_text(values: Iterable[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def normalize_angle(angle: float) -> float:
    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def mirror_pose(values: list[float], axis: str) -> list[float]:
    x, y, z, roll, pitch, yaw = values
    if "x" in axis:
        x = -x
        yaw = normalize_angle(math.pi - yaw)
    if "y" in axis:
        y = -y
        yaw = normalize_angle(-yaw)
    return [x, y, z, roll, pitch, yaw]


def transform_pose_element(parent: ET.Element, axis: str) -> None:
    pose = find_child(parent, "pose")
    if pose is None or pose.text is None:
        return
    pose.text = pose_to_text(mirror_pose(parse_pose(pose.text), axis))


def find_world(root: ET.Element) -> ET.Element:
    world = find_child(root, "world")
    if world is None:
        raise RuntimeError("SDF file does not contain a <world> element.")
    return world


def remove_state_blocks(world: ET.Element) -> int:
    count = 0
    for child in list(world):
        if local_name(child.tag) == "state":
            world.remove(child)
            count += 1
    return count


def is_map_model(element: ET.Element) -> bool:
    if local_name(element.tag) == "model" and element.get("name") == "autonomous_map_3d":
        return True
    if local_name(element.tag) == "include":
        return child_text(element, "uri") == "model://autonomous_map_3d"
    return False


def mirror_top_level_poses(world: ET.Element, axis: str) -> None:
    for child in world:
        if local_name(child.tag) not in {"model", "include", "light"}:
            continue
        if is_map_model(child):
            # The map visual itself is mirrored through its mesh scale; keep its world pose as the anchor.
            continue
        transform_pose_element(child, axis)


def should_remove_model(name: str, mode: str) -> bool:
    if mode == "none":
        return False
    if name.startswith(SIGN_MODEL_PREFIXES):
        return True
    if mode == "signs_and_traffic" and name.startswith(TRAFFIC_MODEL_PREFIXES):
        return True
    return False


def remove_scene_models(world: ET.Element, mode: str) -> list[str]:
    removed: list[str] = []
    for child in list(world):
        if local_name(child.tag) != "model":
            continue
        name = child.get("name", "")
        if should_remove_model(name, mode):
            world.remove(child)
            removed.append(name)
    return removed


def ensure_mesh_scale(mesh: ET.Element, axis: str) -> None:
    scale = find_child(mesh, "scale")
    if scale is None:
        scale = ET.SubElement(mesh, "scale")
    sx = -1.0 if "x" in axis else 1.0
    sy = -1.0 if "y" in axis else 1.0
    scale.text = f"{sx:.1f} {sy:.1f} 1.0"


def mirror_map_visual(world: ET.Element, axis: str) -> int:
    edited = 0
    for model in world.iter():
        if local_name(model.tag) != "model" or model.get("name") != "autonomous_map_3d":
            continue
        for mesh in model.iter():
            if local_name(mesh.tag) == "mesh":
                ensure_mesh_scale(mesh, axis)
                edited += 1
    return edited


def write_baked_mirror_map_model(package_root: Path, axis: str) -> Path:
    source_texture = package_root / "models" / "autonomous_map_3d" / "materials" / "textures" / "autonomous_map_visual.png"
    image = cv2.imread(str(source_texture), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read map texture: {source_texture}")

    if axis == "x":
        mirrored = cv2.flip(image, 1)
    elif axis == "y":
        mirrored = cv2.flip(image, 0)
    else:
        mirrored = cv2.flip(image, -1)

    model_name = f"autonomous_map_3d_mirror_{axis}"
    model_dir = package_root / "models" / model_name
    mesh_dir = model_dir / "meshes"
    texture_dir = model_dir / "materials" / "textures"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    texture_path = texture_dir / "autonomous_map_visual.png"
    mesh_path = mesh_dir / f"{model_name}.dae"
    cv2.imwrite(str(texture_path), mirrored)
    mesh_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>mirrored_world_lab baked mirror map generator</authoring_tool></contributor>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_images>
    <image id="map_texture_image" name="map_texture_image">
      <init_from>../materials/textures/autonomous_map_visual.png</init_from>
    </image>
  </library_images>
  <library_effects>
    <effect id="map_texture_effect">
      <profile_COMMON>
        <newparam sid="map_surface"><surface type="2D"><init_from>map_texture_image</init_from></surface></newparam>
        <newparam sid="map_sampler"><sampler2D><source>map_surface</source></sampler2D></newparam>
        <technique sid="common"><lambert><emission><texture texture="map_sampler" texcoord="UVSET0"/></emission><ambient><color>1.0 1.0 1.0 1</color></ambient><diffuse><texture texture="map_sampler" texcoord="UVSET0"/></diffuse></lambert></technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="map_texture_material" name="map_texture_material"><instance_effect url="#map_texture_effect"/></material>
  </library_materials>
  <library_geometries>
    <geometry id="map_base" name="map_base">
      <mesh>
        <source id="map_base-positions">
          <float_array id="map_base-positions-array" count="12">-3.000000 -3.000000 0.000000 3.000000 -3.000000 0.000000 3.000000 3.000000 0.000000 -3.000000 3.000000 0.000000</float_array>
          <technique_common>
            <accessor source="#map_base-positions-array" count="4" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="map_base-texcoords">
          <float_array id="map_base-texcoords-array" count="8">0.000000 1.000000 1.000000 1.000000 1.000000 0.000000 0.000000 0.000000</float_array>
          <technique_common>
            <accessor source="#map_base-texcoords-array" count="4" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="map_base-vertices">
          <input semantic="POSITION" source="#map_base-positions"/>
        </vertices>
        <triangles material="map_texture" count="2">
          <input semantic="VERTEX" source="#map_base-vertices" offset="0"/>
          <input semantic="TEXCOORD" source="#map_base-texcoords" offset="1" set="0"/>
          <p>0 0 1 1 2 2 0 0 2 2 3 3</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="map_base_node" name="map_base_node">
        <instance_geometry url="#map_base">
          <bind_material><technique_common><instance_material symbol="map_texture" target="#map_texture_material"><bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/></instance_material></technique_common></bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
""",
        encoding="utf-8",
    )
    (model_dir / "model.sdf").write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="map_link">
      <visual name="textured_map_visual">
        <geometry>
          <mesh>
            <uri>{file_uri(mesh_path)}</uri>
          </mesh>
        </geometry>
      </visual>
      <collision name="floor_collision">
        <pose>0 0 -0.01 0 0 0</pose>
        <geometry>
          <box>
            <size>6.0 6.0 0.02</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
""",
        encoding="utf-8",
    )
    (model_dir / "model.config").write_text(
        f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>my_robot_vision</name>
    <email>student@example.com</email>
  </author>
  <description>Mirror-{axis.upper()} baked version of autonomous_map_3d.</description>
</model>
""",
        encoding="utf-8",
    )
    return mesh_path


def use_baked_mirror_map_visual(world: ET.Element, mesh_path: Path, axis: str) -> int:
    edited = 0
    model_name = f"autonomous_map_3d_mirror_{axis}"
    for model in world.iter():
        if local_name(model.tag) != "model" or model.get("name") != "autonomous_map_3d":
            continue
        model.set("name", model_name)
        for mesh in model.iter():
            if local_name(mesh.tag) != "mesh":
                continue
            uri = find_child(mesh, "uri")
            if uri is None:
                uri = ET.SubElement(mesh, "uri")
            uri.text = file_uri(mesh_path)
            scale = find_child(mesh, "scale")
            if scale is not None:
                mesh.remove(scale)
            edited += 1
    return edited


def route_point_to_model(px: float, py: float, offset_x: float = 0.0, offset_y: float = 0.0) -> tuple[float, float]:
    x = (px / MAP_TEXTURE_SIZE - 0.5) * MODEL_SIZE_M + offset_x
    y = (0.5 - py / MAP_TEXTURE_SIZE) * MODEL_SIZE_M + offset_y
    return x, y


def route_anchor_to_px(anchor, nodes: dict[str, list[float]]) -> tuple[float, float]:
    if isinstance(anchor, str):
        point = nodes[anchor]
        return float(point[0]), float(point[1])
    return float(anchor[0]), float(anchor[1])


def flood_outside(whiteish: np.ndarray) -> np.ndarray:
    height, width = whiteish.shape
    outside = np.zeros((height, width), dtype=np.uint8)
    stack = [(0, 0), (width - 1, 0), (0, height - 1), (width - 1, height - 1)]

    while stack:
        x, y = stack.pop()
        if x < 0 or x >= width or y < 0 or y >= height:
            continue
        if outside[y, x] or not whiteish[y, x]:
            continue
        outside[y, x] = 1
        stack.extend(((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)))

    return outside.astype(bool)


def load_map_image(map_image_path: Path) -> np.ndarray:
    image = cv2.imread(str(map_image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read map image: {map_image_path}")
    if image.shape[0] != int(MAP_TEXTURE_SIZE) or image.shape[1] != int(MAP_TEXTURE_SIZE):
        image = cv2.resize(image, (int(MAP_TEXTURE_SIZE), int(MAP_TEXTURE_SIZE)), interpolation=cv2.INTER_NEAREST)
    return image


def build_safe_road_mask(map_image: np.ndarray, clearance_px: int) -> tuple[np.ndarray, np.ndarray]:
    hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, np.array([40, 50, 40], dtype=np.uint8), np.array([95, 255, 255], dtype=np.uint8)) > 0
    yellow = cv2.inRange(hsv, np.array([15, 60, 80], dtype=np.uint8), np.array([45, 255, 255], dtype=np.uint8)) > 0

    whiteish = (hsv[:, :, 1] < 45) & (hsv[:, :, 2] > 180)
    inside_map = ~flood_outside(whiteish)
    blocked = (~inside_map) | green | yellow
    curb_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (CURB_VISUAL_BUFFER_PX * 2 + 1, CURB_VISUAL_BUFFER_PX * 2 + 1),
    )
    blocked = cv2.dilate(blocked.astype(np.uint8), curb_kernel) > 0
    drivable = inside_map & ~blocked

    clearance_px = max(1, int(clearance_px))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (clearance_px * 2 + 1, clearance_px * 2 + 1))
    safe = cv2.erode(drivable.astype(np.uint8), kernel) > 0
    if not np.any(safe):
        raise RuntimeError("Safe road mask is empty. Reduce clearance_px.")

    clearance = cv2.distanceTransform(drivable.astype(np.uint8), cv2.DIST_L2, 5)
    return safe, clearance


def nearest_safe_point(point: tuple[float, float], safe_mask: np.ndarray) -> tuple[int, int]:
    height, width = safe_mask.shape
    x = int(round(min(max(point[0], 0.0), width - 1.0)))
    y = int(round(min(max(point[1], 0.0), height - 1.0)))
    if safe_mask[y, x]:
        return x, y

    safe_y, safe_x = np.nonzero(safe_mask)
    distances = (safe_x - x) * (safe_x - x) + (safe_y - y) * (safe_y - y)
    nearest_index = int(np.argmin(distances))
    return int(safe_x[nearest_index]), int(safe_y[nearest_index])


def astar_path(
    start_point: tuple[float, float],
    goal_point: tuple[float, float],
    safe_mask: np.ndarray,
    clearance: np.ndarray,
) -> list[tuple[int, int]]:
    start = nearest_safe_point(start_point, safe_mask)
    goal = nearest_safe_point(goal_point, safe_mask)
    if start == goal:
        return [start]

    height, width = safe_mask.shape
    open_heap: list[tuple[float, tuple[int, int]]] = [(0.0, start)]
    came_from: dict[tuple[int, int], tuple[int, int]] = {}
    g_score: dict[tuple[int, int], float] = {start: 0.0}
    visited: set[tuple[int, int]] = set()
    neighbors = (
        (-1, 0, 1.0),
        (1, 0, 1.0),
        (0, -1, 1.0),
        (0, 1, 1.0),
        (-1, -1, math.sqrt(2.0)),
        (1, -1, math.sqrt(2.0)),
        (-1, 1, math.sqrt(2.0)),
        (1, 1, math.sqrt(2.0)),
    )

    while open_heap:
        _, current = heapq.heappop(open_heap)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            path.reverse()
            return path

        cx, cy = current
        current_g = g_score[current]
        for dx, dy, step_cost in neighbors:
            nx = cx + dx
            ny = cy + dy
            if nx < 0 or nx >= width or ny < 0 or ny >= height or not safe_mask[ny, nx]:
                continue
            if dx != 0 and dy != 0 and (not safe_mask[cy, nx] or not safe_mask[ny, cx]):
                continue

            center_bias = ROUTE_CENTERLINE_WEIGHT / max(float(clearance[ny, nx]), 1.0)
            tentative_g = current_g + step_cost * (1.0 + center_bias)
            neighbor = (nx, ny)
            if tentative_g >= g_score.get(neighbor, float("inf")):
                continue

            came_from[neighbor] = current
            g_score[neighbor] = tentative_g
            heapq.heappush(open_heap, (tentative_g + math.hypot(float(nx - goal[0]), float(ny - goal[1])), neighbor))

    raise RuntimeError(f"Could not route safely from {start} to {goal}. Reduce clearance_px.")


def line_is_safe(start: tuple[float, float], end: tuple[float, float], clearance: np.ndarray) -> bool:
    height, width = clearance.shape
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    sample_count = max(2, int(math.ceil(max(abs(dx), abs(dy)))) + 1)
    for alpha in np.linspace(0.0, 1.0, sample_count):
        x = int(round(start[0] + dx * float(alpha)))
        y = int(round(start[1] + dy * float(alpha)))
        if x < 0 or x >= width or y < 0 or y >= height or clearance[y, x] <= 0.0:
            return False
    return True


def simplify_straight_route_segment(
    image_points: list[tuple[int, int]],
    start_anchor: tuple[float, float],
    goal_anchor: tuple[float, float],
    clearance: np.ndarray,
) -> list[tuple[float, float]]:
    if len(image_points) < 3:
        return [(float(x), float(y)) for x, y in image_points]

    anchor_dx = goal_anchor[0] - start_anchor[0]
    anchor_dy = goal_anchor[1] - start_anchor[1]
    if min(abs(anchor_dx), abs(anchor_dy)) > STRAIGHT_ROUTE_AXIS_TOLERANCE_PX:
        return [(float(x), float(y)) for x, y in image_points]

    points = np.array(image_points, dtype=np.float32)
    delta = points[-1] - points[0]
    length = float(np.linalg.norm(delta))
    if length < 1.0:
        return [(float(x), float(y)) for x, y in image_points]

    tangent = delta / length
    axis_alignment = max(abs(float(tangent[0])), abs(float(tangent[1])))
    if axis_alignment < 0.985:
        return [(float(x), float(y)) for x, y in image_points]

    normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
    lateral = (points - points[0]) @ normal
    lateral_range = float(np.max(lateral) - np.min(lateral))
    if lateral_range > STRAIGHT_ROUTE_SIMPLIFY_MAX_LATERAL_PX:
        return [(float(x), float(y)) for x, y in image_points]

    start = (float(points[0][0]), float(points[0][1]))
    end = (float(points[-1][0]), float(points[-1][1]))
    if not line_is_safe(start, end, clearance):
        return [(float(x), float(y)) for x, y in image_points]
    return [start, end]


def remove_collinear_points(points: list[tuple[float, float]], epsilon: float = 1e-5) -> list[tuple[float, float]]:
    if len(points) <= 2:
        return points

    simplified = [points[0]]
    for index in range(1, len(points) - 1):
        prev_x, prev_y = simplified[-1]
        x, y = points[index]
        next_x, next_y = points[index + 1]
        cross = (x - prev_x) * (next_y - y) - (y - prev_y) * (next_x - x)
        if abs(cross) > epsilon:
            simplified.append(points[index])
    simplified.append(points[-1])
    return simplified


def route_through_anchors(
    anchors: list[tuple[float, float]],
    safe_mask: np.ndarray,
    clearance: np.ndarray,
) -> list[tuple[float, float]]:
    routed: list[tuple[float, float]] = []
    for index in range(len(anchors) - 1):
        segment = astar_path(anchors[index], anchors[index + 1], safe_mask, clearance)
        segment = simplify_straight_route_segment(segment, anchors[index], anchors[index + 1], clearance)
        if routed:
            segment = segment[1:]
        routed.extend(segment)
    return remove_collinear_points(routed)


def route_polylines(
    config_path: Path,
    map_image_path: Path,
    clearance_px: int,
    planned: bool,
    scope: str,
) -> list[list[tuple[float, float]]]:
    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    route = config["route"]
    nodes = route["nodes"]
    specs = list(route.get("segments", []))
    if scope == "all":
        for direction_specs in route.get("additional_directed_segments", {}).values():
            specs.extend(direction_specs)

    safe_mask = None
    clearance = None
    if planned:
        safe_mask, clearance = build_safe_road_mask(load_map_image(map_image_path), clearance_px)

    polylines: list[list[tuple[float, float]]] = []
    for spec in specs:
        inner_route = bool(spec.get("inner_route", False))
        if scope == "outer" and inner_route:
            continue
        if scope == "inner" and not inner_route:
            continue
        anchors = [route_anchor_to_px(anchor, nodes) for anchor in spec["anchors"]]
        if len(anchors) >= 2:
            if planned:
                assert safe_mask is not None and clearance is not None
                polylines.append(route_through_anchors(anchors, safe_mask, clearance))
            else:
                polylines.append(anchors)
    return polylines


def rgba_material(name: str, ambient: str, diffuse: str, emissive: str) -> ET.Element:
    material = ET.Element("material")
    script = ET.SubElement(material, "script")
    script_name = ET.SubElement(script, "name")
    script_name.text = name
    ambient_el = ET.SubElement(material, "ambient")
    ambient_el.text = ambient
    diffuse_el = ET.SubElement(material, "diffuse")
    diffuse_el.text = diffuse
    emissive_el = ET.SubElement(material, "emissive")
    emissive_el.text = emissive
    return material


def file_uri(path: Path) -> str:
    return path.resolve().as_uri()


def write_route_preview_dae(
    mesh_path: Path,
    polylines: list[list[tuple[float, float]]],
    offset_x: float,
    offset_y: float,
    z: float,
    thickness_m: float,
) -> int:
    positions: list[tuple[float, float, float]] = []
    triangles: list[tuple[int, int, int]] = []
    half_width = thickness_m * 0.5

    for polyline in polylines:
        model_points = [route_point_to_model(px, py, offset_x, offset_y) for px, py in polyline]
        for start, end in zip(model_points, model_points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length < 1e-4:
                continue

            normal_x = -dy / length * half_width
            normal_y = dx / length * half_width
            base = len(positions)
            positions.extend(
                [
                    (start[0] + normal_x, start[1] + normal_y, z),
                    (start[0] - normal_x, start[1] - normal_y, z),
                    (end[0] - normal_x, end[1] - normal_y, z),
                    (end[0] + normal_x, end[1] + normal_y, z),
                ]
            )
            # Duplicate reversed winding so the strip stays visible from both camera sides.
            triangles.extend(
                [
                    (base + 0, base + 1, base + 2),
                    (base + 0, base + 2, base + 3),
                    (base + 2, base + 1, base + 0),
                    (base + 3, base + 2, base + 0),
                ]
            )

    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    position_values = " ".join(f"{value:.6f}" for vertex in positions for value in vertex)
    triangle_values = " ".join(str(index) for triangle in triangles for index in triangle)
    mesh_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>mirrored_world_lab route preview generator</authoring_tool></contributor>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_effects>
    <effect id="route_preview_effect">
      <profile_COMMON>
        <technique sid="common">
          <constant>
            <emission><color>1.0 0.0 1.0 1.0</color></emission>
          </constant>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="route_preview_material" name="route_preview_material">
      <instance_effect url="#route_preview_effect"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="route_preview_geometry" name="route_preview_geometry">
      <mesh>
        <source id="route_preview_positions">
          <float_array id="route_preview_positions_array" count="{len(positions) * 3}">{position_values}</float_array>
          <technique_common>
            <accessor source="#route_preview_positions_array" count="{len(positions)}" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="route_preview_vertices">
          <input semantic="POSITION" source="#route_preview_positions"/>
        </vertices>
        <triangles material="route_preview_material" count="{len(triangles)}">
          <input semantic="VERTEX" source="#route_preview_vertices" offset="0"/>
          <p>{triangle_values}</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="route_preview_node" name="route_preview_node">
        <instance_geometry url="#route_preview_geometry">
          <bind_material>
            <technique_common>
              <instance_material symbol="route_preview_material" target="#route_preview_material"/>
            </technique_common>
          </bind_material>
        </instance_geometry>
      </node>
    </visual_scene>
  </library_visual_scenes>
  <scene><instance_visual_scene url="#Scene"/></scene>
</COLLADA>
""",
        encoding="utf-8",
    )
    return len(triangles)


def make_route_preview_mesh_model(
    name: str,
    mesh_path: Path,
    color_name: str,
    ambient: str,
    diffuse: str,
    emissive: str,
) -> ET.Element:
    model = ET.Element("model", {"name": name})
    static = ET.SubElement(model, "static")
    static.text = "1"
    link = ET.SubElement(model, "link", {"name": "route_preview_link"})
    visual = ET.SubElement(link, "visual", {"name": "route_preview_mesh_visual"})
    geometry = ET.SubElement(visual, "geometry")
    mesh = ET.SubElement(geometry, "mesh")
    uri = ET.SubElement(mesh, "uri")
    uri.text = file_uri(mesh_path)
    visual.append(rgba_material(color_name, ambient, diffuse, emissive))
    pose = ET.SubElement(model, "pose")
    pose.text = "0 0 0 0 0 0"
    return model


def make_route_preview_model(
    name: str,
    polylines: list[list[tuple[float, float]]],
    color_name: str,
    ambient: str,
    diffuse: str,
    emissive: str,
    offset_x: float,
    offset_y: float,
    z: float,
    thickness_m: float,
) -> ET.Element:
    model = ET.Element("model", {"name": name})
    static = ET.SubElement(model, "static")
    static.text = "1"
    link = ET.SubElement(model, "link", {"name": "route_preview_link"})

    visual_index = 0
    for polyline in polylines:
        model_points = [route_point_to_model(px, py, offset_x, offset_y) for px, py in polyline]
        for start, end in zip(model_points, model_points[1:]):
            dx = end[0] - start[0]
            dy = end[1] - start[1]
            length = math.hypot(dx, dy)
            if length < 1e-4:
                continue
            visual = ET.SubElement(link, "visual", {"name": f"route_segment_{visual_index:04d}"})
            visual_index += 1
            pose = ET.SubElement(visual, "pose")
            pose.text = pose_to_text(
                [
                    (start[0] + end[0]) * 0.5,
                    (start[1] + end[1]) * 0.5,
                    z,
                    0.0,
                    0.0,
                    math.atan2(dy, dx),
                ]
            )
            geometry = ET.SubElement(visual, "geometry")
            box = ET.SubElement(geometry, "box")
            size = ET.SubElement(box, "size")
            size.text = f"{length:.6f} {thickness_m:.6f} {thickness_m:.6f}"
            visual.append(rgba_material(color_name, ambient, diffuse, emissive))

    pose = ET.SubElement(model, "pose")
    pose.text = "0 0 0 0 0 0"
    return model


def add_route_previews(
    world: ET.Element,
    config_path: Path,
    map_image_path: Path,
    output_world: Path,
    map_pose: tuple[float, ...],
    mode: str,
    geometry: str,
    planned: bool,
    scope: str,
    clearance_px: int,
) -> None:
    if mode == "none":
        return
    polylines = route_polylines(config_path, map_image_path, clearance_px, planned, scope)
    if mode in {"magenta", "both"}:
        if geometry == "mesh":
            mesh_path = output_world.with_suffix(".route_preview_magenta.dae")
            write_route_preview_dae(
                mesh_path,
                polylines,
                offset_x=0.0,
                offset_y=0.0,
                z=0.040,
                thickness_m=0.020,
            )
            world.append(
                make_route_preview_mesh_model(
                    "route_preview_code_transform_magenta",
                    mesh_path,
                    "Gazebo/RoutePreviewMagenta",
                    "1 0 1 1",
                    "1 0 1 1",
                    "0.7 0 0.7 1",
                )
            )
        else:
            # Magenta: exactly the current code transform, without map visual offset.
            world.append(
                make_route_preview_model(
                    "route_preview_code_transform_magenta",
                    polylines,
                    "Gazebo/RoutePreviewMagenta",
                    "1 0 1 1",
                    "1 0 1 1",
                    "0.7 0 0.7 1",
                    offset_x=0.0,
                    offset_y=0.0,
                    z=0.040,
                    thickness_m=0.020,
                )
            )
    if mode == "both":
        if geometry == "mesh":
            mesh_path = output_world.with_suffix(".route_preview_cyan.dae")
            write_route_preview_dae(
                mesh_path,
                polylines,
                offset_x=float(map_pose[0]),
                offset_y=float(map_pose[1]),
                z=0.055,
                thickness_m=0.014,
            )
            world.append(
                make_route_preview_mesh_model(
                    "route_preview_map_pose_offset_cyan",
                    mesh_path,
                    "Gazebo/RoutePreviewCyan",
                    "0 0.9 1 1",
                    "0 0.9 1 1",
                    "0 0.5 0.8 1",
                )
            )
        else:
            # Cyan: route converted from the 2D texture into the actual map model pose.
            world.append(
                make_route_preview_model(
                    "route_preview_map_pose_offset_cyan",
                    polylines,
                    "Gazebo/RoutePreviewCyan",
                    "0 0.9 1 1",
                    "0 0.9 1 1",
                    "0 0.5 0.8 1",
                    offset_x=float(map_pose[0]),
                    offset_y=float(map_pose[1]),
                    z=0.055,
                    thickness_m=0.014,
                )
            )


def map_pose_from_world(world: ET.Element) -> tuple[float, float, float, float, float, float]:
    for child in world:
        if is_map_model(child):
            pose = parse_pose(child_text(child, "pose"))
            return tuple(pose)  # type: ignore[return-value]
    return DEFAULT_MAP_POSE


def indent_xml(element: ET.Element, level: int = 0) -> None:
    spacer = "\n" + "  " * level
    if len(element):
        if not element.text or not element.text.strip():
            element.text = spacer + "  "
        for child in element:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = spacer
    if level and (not element.tail or not element.tail.strip()):
        element.tail = spacer


def generate_world(args: argparse.Namespace) -> dict:
    package_root = repo_root()
    tree = ET.parse(str(args.source_world))
    root = tree.getroot()
    world = find_world(root)
    map_pose = map_pose_from_world(world)
    removed_states = remove_state_blocks(world)
    removed_models = remove_scene_models(world, args.remove_scene_objects)
    mirror_top_level_poses(world, args.axis)
    baked_map_mesh = None
    if args.bake_mirrored_map:
        baked_map_mesh = write_baked_mirror_map_model(package_root, args.axis)
        mirrored_meshes = use_baked_mirror_map_visual(world, baked_map_mesh, args.axis)
    else:
        mirrored_meshes = mirror_map_visual(world, args.axis)
    add_route_previews(
        world,
        args.route_config,
        args.map_image,
        args.output_world,
        map_pose,
        args.route_preview_mode,
        args.route_preview_geometry,
        args.planned_route_preview,
        args.route_preview_scope,
        args.clearance_px,
    )

    args.output_world.parent.mkdir(parents=True, exist_ok=True)
    indent_xml(root)
    tree.write(str(args.output_world), encoding="utf-8", xml_declaration=True)
    return {
        "source_world": str(args.source_world),
        "output_world": str(args.output_world),
        "axis": args.axis,
        "removed_state_blocks": removed_states,
        "removed_scene_objects": args.remove_scene_objects,
        "removed_models": removed_models,
        "baked_mirrored_map": bool(args.bake_mirrored_map),
        "baked_map_mesh": None if baked_map_mesh is None else str(baked_map_mesh),
        "mirrored_meshes": mirrored_meshes,
        "map_pose": list(map_pose),
        "route_preview": args.route_preview_mode != "none",
        "route_preview_mode": args.route_preview_mode,
        "route_preview_geometry": args.route_preview_geometry,
        "route_preview_planned": bool(args.planned_route_preview),
        "route_preview_scope": args.route_preview_scope,
        "clearance_px": int(args.clearance_px),
    }


def parse_args() -> argparse.Namespace:
    package_root = repo_root()
    default_output = package_root / "mirrored_world_lab" / "generated" / "autonomous_map_3d_mirror_x_route_preview.world"
    default_map_image = package_root / "maps" / "autonomous_map_visual.png"
    parser = argparse.ArgumentParser(description="Generate a mirrored Gazebo world for 2D/3D route alignment testing.")
    parser.add_argument("--source-world", type=Path, default=Path("/home/quocbao/Downloads/map_1"))
    parser.add_argument("--output-world", type=Path, default=default_output)
    parser.add_argument("--route-config", type=Path, default=package_root / "config" / "autonomous_route_graph.json")
    parser.add_argument("--map-image", type=Path, default=default_map_image)
    parser.add_argument("--axis", choices=("x", "y", "xy"), default="x")
    parser.add_argument("--clearance-px", type=int, default=DEFAULT_CLEARANCE_PX)
    parser.add_argument(
        "--remove-scene-objects",
        choices=("none", "signs", "signs_and_traffic"),
        default="none",
        help="Remove generated semantic scene models while keeping the map, car, and world lights.",
    )
    parser.add_argument(
        "--bake-mirrored-map",
        action="store_true",
        help="Create/use a separate mirrored map model asset instead of relying on negative mesh scale.",
    )
    parser.add_argument(
        "--route-preview-mode",
        choices=("both", "magenta", "none"),
        default="both",
        help="Route overlay models to add to the generated world.",
    )
    parser.add_argument(
        "--route-preview-geometry",
        choices=("boxes", "mesh"),
        default="boxes",
        help="Use many box visuals or one lightweight mesh visual for each preview color.",
    )
    parser.add_argument(
        "--route-preview-scope",
        choices=("all", "outer", "inner"),
        default="all",
        help="Which configured route segments to draw in the overlay.",
    )
    parser.add_argument(
        "--anchor-route-preview",
        dest="planned_route_preview",
        action="store_false",
        help="Draw raw config anchors instead of the planned safe-mask route.",
    )
    parser.add_argument("--planned-route-preview", dest="planned_route_preview", action="store_true", default=True)
    parser.add_argument("--add-route-preview", dest="legacy_route_preview", action="store_true", default=None)
    parser.add_argument("--no-route-preview", dest="legacy_route_preview", action="store_false")
    args = parser.parse_args()
    if args.legacy_route_preview is True:
        args.route_preview_mode = "both"
    elif args.legacy_route_preview is False:
        args.route_preview_mode = "none"
    return args


def main() -> int:
    args = parse_args()
    info = generate_world(args)
    info_path = args.output_world.with_suffix(".info.json")
    with info_path.open("w", encoding="utf-8") as handle:
        json.dump(info, handle, indent=2)
    print(f"Generated mirrored world: {args.output_world}")
    print(f"Info: {info_path}")
    print(
        f"axis={info['axis']} mirrored_meshes={info['mirrored_meshes']} "
        f"baked_map={info['baked_mirrored_map']} removed={len(info['removed_models'])} "
        f"route_preview_mode={info['route_preview_mode']} geometry={info['route_preview_geometry']} "
        f"planned={info['route_preview_planned']} scope={info['route_preview_scope']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
