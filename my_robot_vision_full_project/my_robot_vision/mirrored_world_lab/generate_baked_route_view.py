#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

import generate_mirrored_world as mirror


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def clamp_pixel(value: float) -> int:
    return int(round(min(max(value, 0.0), mirror.MAP_TEXTURE_SIZE - 1.0)))


def draw_baked_route(
    map_image_path: Path,
    route_config: Path,
    output_texture: Path,
    clearance_px: int,
) -> dict:
    image = mirror.load_map_image(map_image_path)
    polylines = mirror.route_polylines(
        route_config,
        map_image_path,
        clearance_px,
        planned=True,
        scope="all",
    )

    # The Gazebo map is mirror-X, so bake the route into the opposite texture X.
    # After the mesh scale flips the map, the baked line lands at the same model
    # coordinates as the autonomous code's 2D->3D transform.
    rendered_polylines: list[np.ndarray] = []
    for polyline in polylines:
        points = np.array(
            [
                [
                    clamp_pixel(mirror.MAP_TEXTURE_SIZE - px),
                    clamp_pixel(py),
                ]
                for px, py in polyline
            ],
            dtype=np.int32,
        )
        if len(points) >= 2:
            rendered_polylines.append(points)

    for points in rendered_polylines:
        cv2.polylines(image, [points], False, (20, 20, 20), 6, cv2.LINE_AA)
    for points in rendered_polylines:
        cv2.polylines(image, [points], False, (255, 0, 255), 3, cv2.LINE_AA)

    output_texture.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_texture), image):
        raise RuntimeError(f"Could not write baked texture: {output_texture}")
    return {
        "polylines": len(rendered_polylines),
        "texture": str(output_texture),
    }


def write_baked_map_dae(mesh_path: Path, texture_name: str) -> None:
    mesh_path.parent.mkdir(parents=True, exist_ok=True)
    mesh_path.write_text(
        f"""<?xml version="1.0" encoding="utf-8"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset>
    <contributor><authoring_tool>mirrored_world_lab baked route viewer</authoring_tool></contributor>
    <unit name="meter" meter="1"/>
    <up_axis>Z_UP</up_axis>
  </asset>
  <library_images>
    <image id="map_texture_image" name="map_texture_image">
      <init_from>../materials/textures/{texture_name}</init_from>
    </image>
  </library_images>
  <library_effects>
    <effect id="map_texture_effect">
      <profile_COMMON>
        <newparam sid="map_surface">
          <surface type="2D"><init_from>map_texture_image</init_from></surface>
        </newparam>
        <newparam sid="map_sampler">
          <sampler2D><source>map_surface</source></sampler2D>
        </newparam>
        <technique sid="common">
          <lambert>
            <emission><texture texture="map_sampler" texcoord="UVSET0"/></emission>
            <ambient><color>1.0 1.0 1.0 1</color></ambient>
            <diffuse><texture texture="map_sampler" texcoord="UVSET0"/></diffuse>
          </lambert>
        </technique>
      </profile_COMMON>
    </effect>
  </library_effects>
  <library_materials>
    <material id="map_texture_material" name="map_texture_material">
      <instance_effect url="#map_texture_effect"/>
    </material>
  </library_materials>
  <library_geometries>
    <geometry id="map_base" name="map_base">
      <mesh>
        <source id="map_base_positions">
          <float_array id="map_base_positions_array" count="12">-3 -3 0 3 -3 0 3 3 0 -3 3 0</float_array>
          <technique_common>
            <accessor source="#map_base_positions_array" count="4" stride="3">
              <param name="X" type="float"/>
              <param name="Y" type="float"/>
              <param name="Z" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <source id="map_base_texcoords">
          <float_array id="map_base_texcoords_array" count="8">0 1 1 1 1 0 0 0</float_array>
          <technique_common>
            <accessor source="#map_base_texcoords_array" count="4" stride="2">
              <param name="S" type="float"/>
              <param name="T" type="float"/>
            </accessor>
          </technique_common>
        </source>
        <vertices id="map_base_vertices">
          <input semantic="POSITION" source="#map_base_positions"/>
        </vertices>
        <triangles material="map_texture_material" count="2">
          <input semantic="VERTEX" source="#map_base_vertices" offset="0"/>
          <input semantic="TEXCOORD" source="#map_base_texcoords" offset="1" set="0"/>
          <p>0 0 1 1 2 2 0 0 2 2 3 3</p>
        </triangles>
      </mesh>
    </geometry>
  </library_geometries>
  <library_visual_scenes>
    <visual_scene id="Scene" name="Scene">
      <node id="map_base_node" name="map_base_node">
        <instance_geometry url="#map_base">
          <bind_material>
            <technique_common>
              <instance_material symbol="map_texture_material" target="#map_texture_material">
                <bind_vertex_input semantic="UVSET0" input_semantic="TEXCOORD" input_set="0"/>
              </instance_material>
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


def write_view_world(world_path: Path, mesh_path: Path) -> None:
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.7">
  <world name="baked_route_view_world">
    <scene>
      <ambient>0.85 0.85 0.85 1</ambient>
      <background>0.86 0.91 0.98 1</background>
      <shadows>0</shadows>
    </scene>
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>0 0 7.5 0 1.570796 1.570796</pose>
        <view_controller>orbit</view_controller>
      </camera>
    </gui>
    <light name="sun" type="directional">
      <cast_shadows>0</cast_shadows>
      <pose>0 0 8 0 0 0</pose>
      <diffuse>1 1 1 1</diffuse>
      <specular>0.1 0.1 0.1 1</specular>
      <direction>0 0 -1</direction>
    </light>
    <model name="baked_route_map">
      <static>1</static>
      <pose>0 0.3 0 0 0 0</pose>
      <link name="map_link">
        <visual name="baked_route_visual">
          <geometry>
            <mesh>
              <uri>{mesh_path.resolve().as_uri()}</uri>
              <scale>-1 1 1</scale>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>
  </world>
</sdf>
""",
        encoding="utf-8",
    )


def parse_args() -> argparse.Namespace:
    package_root = repo_root()
    default_dir = package_root / "mirrored_world_lab" / "generated" / "baked_route_view"
    parser = argparse.ArgumentParser(description="Generate a lightweight baked route viewer world.")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--route-config", type=Path, default=package_root / "config" / "autonomous_route_graph.json")
    parser.add_argument("--map-image", type=Path, default=package_root / "maps" / "autonomous_map_visual.png")
    parser.add_argument("--clearance-px", type=int, default=mirror.DEFAULT_CLEARANCE_PX)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    texture_path = args.output_dir / "materials" / "textures" / "autonomous_map_route_baked.png"
    mesh_path = args.output_dir / "meshes" / "autonomous_map_route_baked.dae"
    world_path = args.output_dir / "autonomous_map_3d_mirror_x_route_baked_view.world"
    info_path = args.output_dir / "autonomous_map_3d_mirror_x_route_baked_view.info.json"

    info = draw_baked_route(args.map_image, args.route_config, texture_path, args.clearance_px)
    write_baked_map_dae(mesh_path, texture_path.name)
    write_view_world(world_path, mesh_path)
    info.update(
        {
            "world": str(world_path),
            "mesh": str(mesh_path),
            "route_scope": "all",
            "route_planned": True,
            "route_baked_for_mirror_x": True,
        }
    )
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Generated baked route view world: {world_path}")
    print(f"Generated baked texture: {texture_path}")
    print(f"Info: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
