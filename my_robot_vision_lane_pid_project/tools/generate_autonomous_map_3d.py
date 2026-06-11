from __future__ import annotations

import math
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
MAP_IMAGE = PACKAGE_ROOT / "maps" / "autonomous_map_visual.png"
MODEL_DIR = PACKAGE_ROOT / "models" / "autonomous_map_3d"
MESH_DIR = MODEL_DIR / "meshes"
TEXTURE_DIR = MODEL_DIR / "materials" / "textures"
WORLD_DIR = PACKAGE_ROOT / "worlds"
SIGN_TEXTURES = [
    (
        PACKAGE_ROOT / "go_straight.jpg",
        PACKAGE_ROOT / "models" / "road_sign_go_straight" / "materials" / "textures" / "go_straight.jpg",
    ),
    (
        PACKAGE_ROOT / "limit_speed.jpg",
        PACKAGE_ROOT / "models" / "road_sign_speed_limit" / "materials" / "textures" / "limit_speed.jpg",
    ),
    (
        PACKAGE_ROOT / "parking.png",
        PACKAGE_ROOT / "models" / "road_sign_parking" / "materials" / "textures" / "parking.png",
    ),
    (
        PACKAGE_ROOT / "turn_right.png",
        PACKAGE_ROOT / "models" / "road_sign_turn_right" / "materials" / "textures" / "turn_right.png",
    ),
]

MODEL_NAME = "autonomous_map_3d"
MODEL_SIZE_M = 6.0
MAP_ORIGIN_OFFSET_X_M = 0.0
MAP_ORIGIN_OFFSET_Y_M = 0.30
SCENERY_PROPS = [
    # Keep the scene readable: only a couple of lights, both deep on grass.
    ("traffic_light_toy", "traffic_light_top_island", 104.0, 73.0, 3.1416),
    ("traffic_light_toy", "traffic_light_lower_island", 140.0, 172.0, 0.0),
    # Signs are also set well inside grass islands, not merely on the curb.
    ("road_sign_go_straight", "sign_go_straight_top_island", 70.0, 73.0, 1.5708),
    ("road_sign_turn_right", "sign_turn_right_top_right_island", 186.0, 73.0, 1.5708),
    ("road_sign_speed_limit", "sign_speed_limit_left_island", 68.0, 165.0, -1.5708),
    ("road_sign_parking", "sign_parking_lower_right_island", 189.0, 172.0, 0.0),
]
TRAFFIC_LIGHT_LENS_LOCAL_Y_M = -0.038
TRAFFIC_LIGHT_LENS_Z_M = {
    "red": 0.660,
    "yellow": 0.565,
    "green": 0.470,
}
TRAFFIC_LIGHT_GLOW_INITIAL_STATE = "green"
TRAFFIC_LIGHT_ACTIVE_LAMP_MATERIALS = {
    "red": ("1.0 0.0 0.0 1", "1.0 0.0 0.0 1", "8.0 0.0 0.0 1"),
    "yellow": ("1.0 0.72 0.0 1", "1.0 0.72 0.0 1", "8.0 5.2 0.0 1"),
    "green": ("0.0 1.0 0.05 1", "0.0 1.0 0.05 1", "0.0 8.0 0.30 1"),
}
# Keep the imported Corel map visually faithful. The road, grass, arrows,
# crosswalks, and yellow curbs are preserved by the texture instead of being
# rebuilt as coarse raised blocks.
OBSTACLE_HEIGHT_M = 0.0
LINE_HEIGHT_M = 0.0
LINE_WIDTH_M = 0.18
MIN_SCENERY_GRASS_CLEARANCE_PX = 8.0


def image_to_world(px: float, py: float, width: int, height: int) -> tuple[float, float]:
    x = (px / width - 0.5) * MODEL_SIZE_M
    y = (0.5 - py / height) * MODEL_SIZE_M
    return x, y


def map_pixel_to_world(px: float, py: float) -> tuple[float, float]:
    x, y = image_to_world(px, py, 256, 256)
    return x + MAP_ORIGIN_OFFSET_X_M, y + MAP_ORIGIN_OFFSET_Y_M


def scenery_prop_includes() -> str:
    includes: list[str] = []
    for model_name, instance_name, px, py, yaw in SCENERY_PROPS:
        x, y = map_pixel_to_world(px, py)
        includes.append(
            f"""    <include>
      <uri>model://{model_name}</uri>
      <name>{instance_name}</name>
      <pose>{x:.4f} {y:.4f} 0 0 0 {yaw:.4f}</pose>
    </include>"""
        )
    return "\n".join(includes)


def traffic_light_lens_world_pose(px: float, py: float, yaw: float, z: float) -> tuple[float, float, float]:
    x, y = map_pixel_to_world(px, py)
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    lens_x = x - TRAFFIC_LIGHT_LENS_LOCAL_Y_M * sine
    lens_y = y + TRAFFIC_LIGHT_LENS_LOCAL_Y_M * cosine
    return lens_x, lens_y, z


def traffic_light_indicator_models() -> str:
    models: list[str] = []
    for model_name, instance_name, px, py, yaw in SCENERY_PROPS:
        if model_name != "traffic_light_toy":
            continue

        for color_name, z in TRAFFIC_LIGHT_LENS_Z_M.items():
            ambient, diffuse, emissive = TRAFFIC_LIGHT_ACTIVE_LAMP_MATERIALS[color_name]
            if color_name == TRAFFIC_LIGHT_GLOW_INITIAL_STATE:
                lens_x, lens_y, lens_z = traffic_light_lens_world_pose(px, py, yaw, z)
                pose = f"{lens_x:.4f} {lens_y:.4f} {lens_z:.3f} 0 0 {yaw:.4f}"
            else:
                pose = "0 0 -5 0 0 0"
            models.append(
                f"""    <model name="{instance_name}_{color_name}_active_lamp">
      <static>false</static>
      <link name="lamp_link">
        <gravity>false</gravity>
        <visual name="lamp_visual">
          <geometry>
            <sphere>
              <radius>0.065</radius>
            </sphere>
          </geometry>
          <material>
            <ambient>{ambient}</ambient>
            <diffuse>{diffuse}</diffuse>
            <emissive>{emissive}</emissive>
          </material>
        </visual>
        <kinematic>true</kinematic>
      </link>
      <pose>{pose}</pose>
    </model>"""
            )

    return "\n".join(models)


def validate_scenery_props_on_grass(map_image: np.ndarray) -> None:
    hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([45, 55, 45], dtype=np.uint8), np.array([90, 255, 255], dtype=np.uint8))
    grass_clearance = cv2.distanceTransform((green_mask > 0).astype(np.uint8), cv2.DIST_L2, 5)
    height, width = green_mask.shape[:2]

    failed: list[str] = []
    for _, instance_name, px, py, _ in SCENERY_PROPS:
        x = int(round(px))
        y = int(round(py))
        if x < 0 or x >= width or y < 0 or y >= height:
            failed.append(f"{instance_name} is outside the map at ({px:.1f}, {py:.1f})")
            continue

        clearance = float(grass_clearance[y, x])
        if clearance < MIN_SCENERY_GRASS_CLEARANCE_PX:
            failed.append(
                f"{instance_name} is too close to road/curb at ({px:.1f}, {py:.1f}), "
                f"grass clearance={clearance:.1f}px"
            )

    if failed:
        details = "\n  - ".join(failed)
        raise RuntimeError(f"Invalid scenery prop placement:\n  - {details}")


def world_layout_to_model(point: np.ndarray) -> tuple[float, float]:
    sys.path.insert(0, str(PACKAGE_ROOT))
    from my_robot_vision.world import autonomous_map_layout

    (x0, y0, map_size), _, _ = autonomous_map_layout()
    px = (float(point[0]) - x0) / map_size * 256.0
    py = (float(point[1]) - y0) / map_size * 256.0
    return image_to_world(px, py, 256, 256)


def add_quad(vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]], quad: list[tuple[float, float, float]]) -> None:
    start = len(vertices)
    vertices.extend(quad)
    faces.append((start, start + 1, start + 2))
    faces.append((start, start + 2, start + 3))


def build_obstacle_mesh(map_image: np.ndarray, grid_size: int = 112) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    hsv = cv2.cvtColor(map_image, cv2.COLOR_BGR2HSV)
    green_mask = cv2.inRange(hsv, np.array([45, 55, 45], dtype=np.uint8), np.array([90, 255, 255], dtype=np.uint8))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    occupied = cv2.resize(green_mask, (grid_size, grid_size), interpolation=cv2.INTER_AREA) > 60

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    cell = MODEL_SIZE_M / grid_size
    z0 = 0.0
    z1 = OBSTACLE_HEIGHT_M

    def cell_bounds(row: int, col: int) -> tuple[float, float, float, float]:
        x_left = -MODEL_SIZE_M / 2.0 + col * cell
        x_right = x_left + cell
        y_top = MODEL_SIZE_M / 2.0 - row * cell
        y_bottom = y_top - cell
        return x_left, x_right, y_bottom, y_top

    for row in range(grid_size):
        for col in range(grid_size):
            if not occupied[row, col]:
                continue

            x_left, x_right, y_bottom, y_top = cell_bounds(row, col)
            add_quad(vertices, faces, [(x_left, y_bottom, z1), (x_right, y_bottom, z1), (x_right, y_top, z1), (x_left, y_top, z1)])

            if row == 0 or not occupied[row - 1, col]:
                add_quad(vertices, faces, [(x_left, y_top, z0), (x_right, y_top, z0), (x_right, y_top, z1), (x_left, y_top, z1)])
            if row == grid_size - 1 or not occupied[row + 1, col]:
                add_quad(vertices, faces, [(x_right, y_bottom, z0), (x_left, y_bottom, z0), (x_left, y_bottom, z1), (x_right, y_bottom, z1)])
            if col == 0 or not occupied[row, col - 1]:
                add_quad(vertices, faces, [(x_left, y_bottom, z0), (x_left, y_top, z0), (x_left, y_top, z1), (x_left, y_bottom, z1)])
            if col == grid_size - 1 or not occupied[row, col + 1]:
                add_quad(vertices, faces, [(x_right, y_top, z0), (x_right, y_bottom, z0), (x_right, y_bottom, z1), (x_right, y_top, z1)])

    return vertices, faces


def build_line_mesh() -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    sys.path.insert(0, str(PACKAGE_ROOT))
    from my_robot_vision.world import build_autonomous_map_track_points

    points = build_autonomous_map_track_points().astype(np.float32)
    model_points = np.array([world_layout_to_model(point) for point in points], dtype=np.float32)
    half_width = LINE_WIDTH_M / 2.0
    z = LINE_HEIGHT_M

    left_vertices: list[tuple[float, float, float]] = []
    right_vertices: list[tuple[float, float, float]] = []

    for index, point in enumerate(model_points):
        prev_point = model_points[index - 1]
        next_point = model_points[(index + 1) % len(model_points)]
        tangent = next_point - prev_point
        norm = float(np.linalg.norm(tangent))
        if norm < 1e-6:
            tangent = np.array([1.0, 0.0], dtype=np.float32)
        else:
            tangent = tangent / norm
        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        left = point + normal * half_width
        right = point - normal * half_width
        left_vertices.append((float(left[0]), float(left[1]), z))
        right_vertices.append((float(right[0]), float(right[1]), z))

    vertices = left_vertices + right_vertices
    faces: list[tuple[int, int, int]] = []
    count = len(model_points)
    for index in range(count):
        nxt = (index + 1) % count
        faces.append((index, nxt, count + nxt))
        faces.append((index, count + nxt, count + index))
    return vertices, faces


def float_list(values: list[float]) -> str:
    return " ".join(f"{value:.6f}" for value in values)


def int_list(values: list[int]) -> str:
    return " ".join(str(value) for value in values)


def geometry_xml(geometry_id: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, int, int]], material_symbol: str) -> str:
    positions = [coord for vertex in vertices for coord in vertex]
    face_indices = [index for face in faces for index in face]
    return f"""
      <geometry id="{geometry_id}" name="{geometry_id}">
        <mesh>
          <source id="{geometry_id}-positions">
            <float_array id="{geometry_id}-positions-array" count="{len(positions)}">{float_list(positions)}</float_array>
            <technique_common>
              <accessor source="#{geometry_id}-positions-array" count="{len(vertices)}" stride="3">
                <param name="X" type="float"/>
                <param name="Y" type="float"/>
                <param name="Z" type="float"/>
              </accessor>
            </technique_common>
          </source>
          <vertices id="{geometry_id}-vertices">
            <input semantic="POSITION" source="#{geometry_id}-positions"/>
          </vertices>
          <triangles material="{material_symbol}" count="{len(faces)}">
            <input semantic="VERTEX" source="#{geometry_id}-vertices" offset="0"/>
            <p>{int_list(face_indices)}</p>
          </triangles>
        </mesh>
      </geometry>"""


def textured_base_geometry() -> str:
    half = MODEL_SIZE_M / 2.0
    positions = [
        -half,
        -half,
        0.0,
        half,
        -half,
        0.0,
        half,
        half,
        0.0,
        -half,
        half,
        0.0,
    ]
    texcoords = [0.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0]
    p = [0, 0, 1, 1, 2, 2, 0, 0, 2, 2, 3, 3]
    return f"""
      <geometry id="map_base" name="map_base">
        <mesh>
          <source id="map_base-positions">
            <float_array id="map_base-positions-array" count="12">{float_list(positions)}</float_array>
            <technique_common>
              <accessor source="#map_base-positions-array" count="4" stride="3">
                <param name="X" type="float"/>
                <param name="Y" type="float"/>
                <param name="Z" type="float"/>
              </accessor>
            </technique_common>
          </source>
          <source id="map_base-texcoords">
            <float_array id="map_base-texcoords-array" count="8">{float_list(texcoords)}</float_array>
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
            <p>{int_list(p)}</p>
          </triangles>
        </mesh>
      </geometry>"""


def write_dae() -> None:
    mesh_path = MESH_DIR / "autonomous_map_3d.dae"
    dae = f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>my_robot_vision map generator</authoring_tool></contributor>
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
{textured_base_geometry()}
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
"""
    mesh_path.write_text(dae, encoding="utf-8")


def write_model_files() -> None:
    (MODEL_DIR / "model.config").write_text(
        f"""<?xml version="1.0"?>
<model>
  <name>{MODEL_NAME}</name>
  <version>1.0</version>
  <sdf version="1.6">model.sdf</sdf>
  <author>
    <name>my_robot_vision</name>
  </author>
  <description>Textured 3D autonomous driving map generated from autonomous_map_in_corel.cdr.</description>
</model>
""",
        encoding="utf-8",
    )

    (MODEL_DIR / "model.sdf").write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.6">
  <model name="{MODEL_NAME}">
    <static>true</static>
    <link name="map_link">
      <visual name="textured_map_visual">
        <geometry>
          <mesh>
            <uri>model://{MODEL_NAME}/meshes/autonomous_map_3d.dae</uri>
          </mesh>
        </geometry>
      </visual>
      <collision name="floor_collision">
        <pose>0 0 -0.01 0 0 0</pose>
        <geometry>
          <box>
            <size>{MODEL_SIZE_M} {MODEL_SIZE_M} 0.02</size>
          </box>
        </geometry>
      </collision>
    </link>
  </model>
</sdf>
""",
        encoding="utf-8",
    )

    WORLD_DIR.mkdir(parents=True, exist_ok=True)
    (WORLD_DIR / "autonomous_map_3d.world").write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.6">
  <world name="autonomous_map_3d_world">
    <plugin name="gazebo_ros_state" filename="libgazebo_ros_state.so">
      <update_rate>50.0</update_rate>
    </plugin>
    <scene>
      <ambient>0.82 0.82 0.82 1</ambient>
      <background>0.86 0.91 0.98 1</background>
      <shadows>false</shadows>
    </scene>
    <include>
      <uri>model://sun</uri>
    </include>
    <light name="key_light" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 6 0 0 0</pose>
      <diffuse>0.92 0.92 0.90 1</diffuse>
      <specular>0.15 0.15 0.15 1</specular>
      <direction>-0.30 0.35 -1.0</direction>
    </light>
    <light name="fill_light" type="directional">
      <cast_shadows>false</cast_shadows>
      <pose>0 0 6 0 0 0</pose>
      <diffuse>0.42 0.46 0.52 1</diffuse>
      <specular>0.05 0.05 0.05 1</specular>
      <direction>0.30 -0.20 -1.0</direction>
    </light>
    <include>
      <uri>model://{MODEL_NAME}</uri>
      <pose>{MAP_ORIGIN_OFFSET_X_M} {MAP_ORIGIN_OFFSET_Y_M} 0 0 0 0</pose>
    </include>
{scenery_prop_includes()}
{traffic_light_indicator_models()}
    <include>
      <uri>model://track_preview_car</uri>
      <name>track_preview_car</name>
      <pose>-1.04 1.44 0.0 0 0 0</pose>
    </include>
    <physics name="default_physics" default="true" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_update_rate>1000</real_time_update_rate>
    </physics>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 -4.7 4.2 0 0.78 1.5708</pose>
      </camera>
    </gui>
  </world>
</sdf>
""",
        encoding="utf-8",
    )


def copy_sign_textures() -> None:
    for source_path, target_path in SIGN_TEXTURES:
        if not source_path.exists():
            raise FileNotFoundError(f"Missing sign texture: {source_path}")
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)


def main() -> None:
    if not MAP_IMAGE.exists():
        raise FileNotFoundError(f"Missing map image: {MAP_IMAGE}")

    MESH_DIR.mkdir(parents=True, exist_ok=True)
    TEXTURE_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(MAP_IMAGE, TEXTURE_DIR / "autonomous_map_visual.png")

    map_image = cv2.imread(str(MAP_IMAGE), cv2.IMREAD_COLOR)
    if map_image is None:
        raise RuntimeError(f"Could not read {MAP_IMAGE}")

    validate_scenery_props_on_grass(map_image)
    write_dae()
    copy_sign_textures()
    write_model_files()

    print(f"Generated {MODEL_NAME}")
    print(f"  mesh: {MESH_DIR / 'autonomous_map_3d.dae'}")
    print(f"  world: {WORLD_DIR / 'autonomous_map_3d.world'}")
    print("  mesh style: faithful flat Corel texture")


if __name__ == "__main__":
    main()
