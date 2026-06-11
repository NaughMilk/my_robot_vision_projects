#!/usr/bin/env python3
"""Build Gazebo 3D road signs from 2D map coordinates and sign images.

This is a standalone reconstruction of the sign workflow used in this project:

1. A sign is authored in 2D map/image coordinates.
2. Pixel coordinates are converted to Gazebo model coordinates.
3. A small static SDF sign model is generated or reused.
4. The sign is inserted into a Gazebo world with a pose and yaw.

The script intentionally uses only Python stdlib so it is easy to copy to
another ROS/Gazebo project.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


MAP_TEXTURE_SIZE_PX = 256.0
MODEL_SIZE_M = 6.0
MAP_VISUAL_OFFSET_X_M = 0.0
MAP_VISUAL_OFFSET_Y_M = 0.3
SIGN_FACE_LOCAL_YAW_OFFSET_RAD = -math.pi * 0.5


@dataclass(frozen=True)
class SignKind:
    key: str
    model_name: str
    texture_name: str | None
    material_name: str | None
    maneuver: str | None


@dataclass(frozen=True)
class SignPlacement:
    name: str
    kind: str
    px: float
    py: float
    yaw_deg: float
    z_m: float = 0.0


SIGN_KINDS: dict[str, SignKind] = {
    "go_straight": SignKind(
        "go_straight",
        "road_sign_go_straight",
        "go_straight.jpg",
        "RoadSign/GoStraight",
        "straight",
    ),
    "turn_left": SignKind(
        "turn_left",
        "road_sign_turn_left",
        "turn_left.png",
        "RoadSign/TurnLeft",
        "left",
    ),
    "turn_right": SignKind(
        "turn_right",
        "road_sign_turn_right",
        "turn_right.png",
        "RoadSign/TurnRight",
        "right",
    ),
    "parking": SignKind(
        "parking",
        "road_sign_parking",
        "parking.png",
        "RoadSign/Parking",
        "parking",
    ),
    "speed_limit": SignKind(
        "speed_limit",
        "road_sign_speed_limit",
        "limit_speed.jpg",
        "RoadSign/SpeedLimit",
        None,
    ),
    "stop": SignKind("stop", "road_sign_stop", None, None, None),
    "arrow": SignKind("arrow", "road_sign_arrow", None, None, None),
}


def map_point_to_model(
    px: float,
    py: float,
    texture_size_px: float = MAP_TEXTURE_SIZE_PX,
    model_size_m: float = MODEL_SIZE_M,
    map_offset_x_m: float = MAP_VISUAL_OFFSET_X_M,
    map_offset_y_m: float = MAP_VISUAL_OFFSET_Y_M,
) -> tuple[float, float]:
    """Convert 2D image pixel coordinates to Gazebo XY coordinates.

    Pixel origin is top-left. Gazebo model origin is map center.
    X grows right in both systems. Y is flipped because image Y grows down.
    """

    model_x = (px / texture_size_px - 0.5) * model_size_m + map_offset_x_m
    model_y = (0.5 - py / texture_size_px) * model_size_m + map_offset_y_m
    return model_x, model_y


def model_point_to_map(
    x_m: float,
    y_m: float,
    texture_size_px: float = MAP_TEXTURE_SIZE_PX,
    model_size_m: float = MODEL_SIZE_M,
    map_offset_x_m: float = MAP_VISUAL_OFFSET_X_M,
    map_offset_y_m: float = MAP_VISUAL_OFFSET_Y_M,
) -> tuple[float, float]:
    """Inverse of map_point_to_model."""

    local_x = x_m - map_offset_x_m
    local_y = y_m - map_offset_y_m
    px = (local_x / model_size_m + 0.5) * texture_size_px
    py = (0.5 - local_y / model_size_m) * texture_size_px
    return px, py


def model_yaw_from_face_yaw(face_yaw_rad: float) -> float:
    """Convert desired sign-face direction to model yaw.

    The sign plate front in these SDF models points along local -Y, so the face
    direction is model_yaw - pi/2. Therefore model_yaw = face_yaw + pi/2.
    """

    return face_yaw_rad - SIGN_FACE_LOCAL_YAW_OFFSET_RAD


def normalize_yaw(yaw_rad: float) -> float:
    return math.atan2(math.sin(yaw_rad), math.cos(yaw_rad))


def pose_text(x: float, y: float, z: float, yaw_rad: float) -> str:
    return f"{x:.6f} {y:.6f} {z:.6f} 0.000000 -0.000000 {normalize_yaw(yaw_rad):.6f}"


def safe_model_token(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in value.lower())


def make_material_script(material_name: str, texture_name: str) -> str:
    return f"""material {material_name}
{{
  technique
  {{
    pass
    {{
      lighting on
      texture_unit
      {{
        texture {texture_name}
      }}
    }}
  }}
}}
"""


def make_model_config(model_name: str) -> str:
    pretty_name = model_name.replace("_", " ").title()
    return f"""<?xml version="1.0"?>
<model>
  <name>{model_name}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>Generated</name>
    <email>generated@example.com</email>
  </author>
  <description>{pretty_name} generated from a 2D sign texture.</description>
</model>
"""


def make_textured_sign_sdf(model_name: str, material_name: str) -> str:
    return f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{model_name}">
    <static>true</static>
    <link name="sign_link">
      <visual name="base_visual">
        <pose>0 0 0.010 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.055</radius>
            <length>0.020</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.72 0.72 0.70 1</ambient>
          <diffuse>0.78 0.78 0.76 1</diffuse>
        </material>
      </visual>
      <visual name="pole_visual">
        <pose>0 0 0.245 0 0 0</pose>
        <geometry>
          <cylinder>
            <radius>0.011</radius>
            <length>0.470</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.72 0.72 0.70 1</ambient>
          <diffuse>0.82 0.82 0.80 1</diffuse>
        </material>
      </visual>
      <visual name="back_plate_visual">
        <pose>0 0 0.520 0 0 0</pose>
        <geometry>
          <box>
            <size>0.260 0.014 0.260</size>
          </box>
        </geometry>
        <material>
          <ambient>0.96 0.96 0.94 1</ambient>
          <diffuse>0.98 0.98 0.96 1</diffuse>
        </material>
      </visual>
      <visual name="sign_face_visual">
        <pose>0 -0.011 0.520 0 0 0</pose>
        <geometry>
          <box>
            <size>0.246 0.004 0.246</size>
          </box>
        </geometry>
        <material>
          <script>
            <uri>model://{model_name}/materials/scripts</uri>
            <uri>model://{model_name}/materials/textures</uri>
            <name>{material_name}</name>
          </script>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""


def make_stop_sign_sdf() -> str:
    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="road_sign_stop">
    <static>true</static>
    <link name="stop_sign_link">
      <visual name="base_visual">
        <pose>0 0 0.010 0 0 0</pose>
        <geometry><cylinder><radius>0.060</radius><length>0.020</length></cylinder></geometry>
        <material><ambient>0.72 0.72 0.70 1</ambient><diffuse>0.78 0.78 0.76 1</diffuse></material>
      </visual>
      <visual name="pole_visual">
        <pose>0 0 0.245 0 0 0</pose>
        <geometry><cylinder><radius>0.011</radius><length>0.470</length></cylinder></geometry>
        <material><ambient>0.72 0.72 0.70 1</ambient><diffuse>0.82 0.82 0.80 1</diffuse></material>
      </visual>
      <visual name="white_back_plate_visual">
        <pose>0 -0.002 0.520 1.5708 0 0</pose>
        <geometry><cylinder><radius>0.108</radius><length>0.012</length></cylinder></geometry>
        <material><ambient>1.0 1.0 1.0 1</ambient><diffuse>1.0 1.0 1.0 1</diffuse></material>
      </visual>
      <visual name="red_plate_visual">
        <pose>0 -0.010 0.520 1.5708 0 0</pose>
        <geometry><cylinder><radius>0.094</radius><length>0.014</length></cylinder></geometry>
        <material><ambient>0.72 0.02 0.02 1</ambient><diffuse>0.95 0.04 0.04 1</diffuse></material>
      </visual>
      <visual name="stop_bar_visual">
        <pose>0 -0.020 0.520 0 0 0</pose>
        <geometry><box><size>0.110 0.006 0.028</size></box></geometry>
        <material><ambient>1.0 1.0 1.0 1</ambient><diffuse>1.0 1.0 1.0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""


def make_arrow_sign_sdf() -> str:
    return """<?xml version="1.0"?>
<sdf version="1.6">
  <model name="road_sign_arrow">
    <static>true</static>
    <link name="arrow_sign_link">
      <visual name="base_visual">
        <pose>0 0 0.010 0 0 0</pose>
        <geometry><cylinder><radius>0.055</radius><length>0.020</length></cylinder></geometry>
        <material><ambient>0.72 0.72 0.70 1</ambient><diffuse>0.78 0.78 0.76 1</diffuse></material>
      </visual>
      <visual name="pole_visual">
        <pose>0 0 0.230 0 0 0</pose>
        <geometry><cylinder><radius>0.010</radius><length>0.440</length></cylinder></geometry>
        <material><ambient>0.72 0.72 0.70 1</ambient><diffuse>0.82 0.82 0.80 1</diffuse></material>
      </visual>
      <visual name="blue_plate_visual">
        <pose>0 -0.004 0.500 1.5708 0 0</pose>
        <geometry><cylinder><radius>0.102</radius><length>0.014</length></cylinder></geometry>
        <material><ambient>0.02 0.18 0.68 1</ambient><diffuse>0.03 0.24 0.92 1</diffuse></material>
      </visual>
      <visual name="arrow_stem_visual">
        <pose>0 -0.018 0.475 0 0 0</pose>
        <geometry><box><size>0.030 0.006 0.105</size></box></geometry>
        <material><ambient>1.0 1.0 1.0 1</ambient><diffuse>1.0 1.0 1.0 1</diffuse></material>
      </visual>
      <visual name="arrow_head_left_visual">
        <pose>-0.028 -0.020 0.545 0 -0.75 0</pose>
        <geometry><box><size>0.020 0.006 0.075</size></box></geometry>
        <material><ambient>1.0 1.0 1.0 1</ambient><diffuse>1.0 1.0 1.0 1</diffuse></material>
      </visual>
      <visual name="arrow_head_right_visual">
        <pose>0.028 -0.020 0.545 0 0.75 0</pose>
        <geometry><box><size>0.020 0.006 0.075</size></box></geometry>
        <material><ambient>1.0 1.0 1.0 1</ambient><diffuse>1.0 1.0 1.0 1</diffuse></material>
      </visual>
    </link>
  </model>
</sdf>
"""


def write_textured_sign_model(
    models_dir: Path,
    kind: SignKind,
    texture_dir: Path,
    overwrite: bool,
) -> None:
    if kind.texture_name is None or kind.material_name is None:
        raise ValueError(f"{kind.key} is not a textured sign kind")

    model_dir = models_dir / kind.model_name
    texture_src = texture_dir / kind.texture_name
    texture_dst = model_dir / "materials" / "textures" / kind.texture_name
    material_dst = (
        model_dir
        / "materials"
        / "scripts"
        / f"{kind.model_name}.material"
    )
    sdf_dst = model_dir / "model.sdf"
    config_dst = model_dir / "model.config"

    if not texture_src.is_file():
        raise FileNotFoundError(f"Missing texture for {kind.key}: {texture_src}")
    if model_dir.exists() and not overwrite:
        return

    texture_dst.parent.mkdir(parents=True, exist_ok=True)
    material_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(texture_src, texture_dst)
    material_dst.write_text(
        make_material_script(kind.material_name, kind.texture_name),
        encoding="utf-8",
    )
    sdf_dst.write_text(
        make_textured_sign_sdf(kind.model_name, kind.material_name),
        encoding="utf-8",
    )
    config_dst.write_text(make_model_config(kind.model_name), encoding="utf-8")


def write_procedural_model(models_dir: Path, kind: SignKind, overwrite: bool) -> None:
    model_dir = models_dir / kind.model_name
    if model_dir.exists() and not overwrite:
        return

    model_dir.mkdir(parents=True, exist_ok=True)
    if kind.key == "stop":
        sdf_text = make_stop_sign_sdf()
    elif kind.key == "arrow":
        sdf_text = make_arrow_sign_sdf()
    else:
        raise ValueError(f"No procedural model generator for {kind.key}")

    (model_dir / "model.sdf").write_text(sdf_text, encoding="utf-8")
    (model_dir / "model.config").write_text(
        make_model_config(kind.model_name),
        encoding="utf-8",
    )


def write_sign_models(
    models_dir: Path,
    texture_dir: Path,
    include_kinds: set[str],
    overwrite: bool = False,
) -> None:
    models_dir.mkdir(parents=True, exist_ok=True)
    for kind_key in sorted(include_kinds):
        if kind_key not in SIGN_KINDS:
            raise ValueError(f"Unknown sign kind: {kind_key}")
        kind = SIGN_KINDS[kind_key]
        if kind.texture_name is None:
            write_procedural_model(models_dir, kind, overwrite)
        else:
            write_textured_sign_model(models_dir, kind, texture_dir, overwrite)


def read_signs(path: Path) -> list[SignPlacement]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("signs JSON must be a list")

    signs: list[SignPlacement] = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(f"signs[{index}] must be an object")
        signs.append(
            SignPlacement(
                name=str(item.get("name") or f"sign_{index:02d}"),
                kind=str(item["kind"]),
                px=float(item["px"]),
                py=float(item["py"]),
                yaw_deg=float(item.get("yaw_deg", 0.0)),
                z_m=float(item.get("z_m", 0.0)),
            )
        )
    return signs


def make_include_element(
    sign: SignPlacement,
    texture_size_px: float,
    model_size_m: float,
    map_offset_x_m: float,
    map_offset_y_m: float,
) -> ET.Element:
    if sign.kind not in SIGN_KINDS:
        raise ValueError(f"Unknown sign kind: {sign.kind}")
    kind = SIGN_KINDS[sign.kind]

    x, y = map_point_to_model(
        sign.px,
        sign.py,
        texture_size_px=texture_size_px,
        model_size_m=model_size_m,
        map_offset_x_m=map_offset_x_m,
        map_offset_y_m=map_offset_y_m,
    )
    yaw_rad = math.radians(sign.yaw_deg)

    include = ET.Element("include")
    name = ET.SubElement(include, "name")
    name.text = safe_model_token(sign.name)
    uri = ET.SubElement(include, "uri")
    uri.text = f"model://{kind.model_name}"
    pose = ET.SubElement(include, "pose")
    pose.text = pose_text(x, y, sign.z_m, yaw_rad)
    return include


def indent_xml(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print ElementTree in-place."""

    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent_xml(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def insert_includes_into_world(
    input_world: Path,
    output_world: Path,
    includes: list[ET.Element],
    remove_name_prefixes: tuple[str, ...] = ("palette_",),
) -> None:
    tree = ET.parse(input_world)
    root = tree.getroot()
    world = root.find("world")
    if world is None:
        raise ValueError(f"No <world> element in {input_world}")

    # Remove old generated sign includes/models if requested.
    for child in list(world):
        name = child.findtext("name")
        attr_name = child.attrib.get("name")
        candidate_name = name or attr_name or ""
        if any(candidate_name.startswith(prefix) for prefix in remove_name_prefixes):
            world.remove(child)

    insert_index = len(world)
    for index, child in enumerate(list(world)):
        if child.tag in {"physics", "gui", "state"}:
            insert_index = index
            break

    for offset, include in enumerate(includes):
        world.insert(insert_index + offset, include)

    indent_xml(root)
    output_world.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output_world, encoding="utf-8", xml_declaration=True)


def write_example_signs(path: Path) -> None:
    example = [
        {"name": "palette_turn_right_03_yaw_270", "kind": "turn_right", "px": 43.0, "py": 126.0, "yaw_deg": 270},
        {"name": "palette_parking_00_yaw_000", "kind": "parking", "px": 156.0, "py": 103.0, "yaw_deg": 0},
        {"name": "palette_speed_limit_01_yaw_090", "kind": "speed_limit", "px": 208.0, "py": 185.0, "yaw_deg": 90},
        {"name": "palette_stop_02_yaw_180", "kind": "stop", "px": 110.0, "py": 214.0, "yaw_deg": 180},
    ]
    path.write_text(json.dumps(example, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-world", type=Path, help="Base Gazebo world to patch.")
    parser.add_argument("--output-world", type=Path, help="Output Gazebo world.")
    parser.add_argument("--signs-json", type=Path, help="JSON list of sign placements in 2D pixels.")
    parser.add_argument("--models-dir", type=Path, default=Path("models"), help="Gazebo models output dir.")
    parser.add_argument("--texture-dir", type=Path, default=Path("sign_textures"), help="Directory containing 2D sign images.")
    parser.add_argument("--texture-size-px", type=float, default=MAP_TEXTURE_SIZE_PX)
    parser.add_argument("--model-size-m", type=float, default=MODEL_SIZE_M)
    parser.add_argument("--map-offset-x-m", type=float, default=MAP_VISUAL_OFFSET_X_M)
    parser.add_argument("--map-offset-y-m", type=float, default=MAP_VISUAL_OFFSET_Y_M)
    parser.add_argument("--write-example-signs", type=Path, help="Write an example signs JSON and exit.")
    parser.add_argument("--create-models", action="store_true", help="Generate model folders under --models-dir.")
    parser.add_argument("--overwrite-models", action="store_true", help="Overwrite generated model folders.")
    args = parser.parse_args()

    if args.write_example_signs is not None:
        write_example_signs(args.write_example_signs)
        print(f"Wrote example signs JSON: {args.write_example_signs}")
        return

    if args.input_world is None or args.output_world is None or args.signs_json is None:
        parser.error("--input-world, --output-world, and --signs-json are required unless --write-example-signs is used")

    signs = read_signs(args.signs_json)
    used_kinds = {sign.kind for sign in signs}

    if args.create_models:
        write_sign_models(
            args.models_dir,
            args.texture_dir,
            used_kinds,
            overwrite=args.overwrite_models,
        )

    includes = [
        make_include_element(
            sign,
            texture_size_px=args.texture_size_px,
            model_size_m=args.model_size_m,
            map_offset_x_m=args.map_offset_x_m,
            map_offset_y_m=args.map_offset_y_m,
        )
        for sign in signs
    ]
    insert_includes_into_world(args.input_world, args.output_world, includes)

    print(f"Wrote world: {args.output_world}")
    print(f"Inserted {len(includes)} sign include(s)")
    if args.create_models:
        print(f"Generated/reused models in: {args.models_dir}")


if __name__ == "__main__":
    main()
