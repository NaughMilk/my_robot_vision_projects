#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2

import generate_mirrored_world as mirror


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def write_world(world_path: Path, map_mesh_path: Path, route_mesh_path: Path) -> None:
    world_path.parent.mkdir(parents=True, exist_ok=True)
    world_path.write_text(
        f"""<?xml version="1.0"?>
<sdf version="1.7">
  <world name="mirror_x_overlay_view_world">
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
    <model name="autonomous_map_3d_mirror_x_view">
      <static>1</static>
      <pose>0 0.3 0 0 0 0</pose>
      <link name="map_link">
        <visual name="mirrored_map_visual">
          <geometry>
            <mesh>
              <uri>{map_mesh_path.resolve().as_uri()}</uri>
              <scale>-1 1 1</scale>
            </mesh>
          </geometry>
        </visual>
      </link>
    </model>
    <model name="route_overlay_code_transform_magenta">
      <static>1</static>
      <pose>0 0 0 0 0 0</pose>
      <link name="route_overlay_link">
        <visual name="route_overlay_mesh_visual">
          <geometry>
            <mesh>
              <uri>{route_mesh_path.resolve().as_uri()}</uri>
            </mesh>
          </geometry>
          <material>
            <ambient>1 0 1 1</ambient>
            <diffuse>1 0 1 1</diffuse>
            <emissive>0.7 0 0.7 1</emissive>
          </material>
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
    default_dir = package_root / "mirrored_world_lab" / "generated" / "mirror_overlay_view"
    parser = argparse.ArgumentParser(description="Generate a lightweight mirror-X map with route overlay view.")
    parser.add_argument("--output-dir", type=Path, default=default_dir)
    parser.add_argument("--route-config", type=Path, default=package_root / "config" / "autonomous_route_graph.json")
    parser.add_argument("--map-image", type=Path, default=package_root / "maps" / "autonomous_map_visual.png")
    parser.add_argument("--map-mesh", type=Path, default=package_root / "models" / "autonomous_map_3d" / "meshes" / "autonomous_map_3d.dae")
    parser.add_argument("--clearance-px", type=int, default=mirror.DEFAULT_CLEARANCE_PX)
    parser.add_argument("--thickness-m", type=float, default=0.026)
    parser.add_argument("--route-mirror-x", action="store_true")
    parser.add_argument("--plan-on-mirror-x", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    world_path = args.output_dir / "autonomous_map_3d_mirror_x_route_overlay_view.world"
    route_mesh_path = args.output_dir / "route_overlay_code_transform_magenta.dae"
    info_path = args.output_dir / "autonomous_map_3d_mirror_x_route_overlay_view.info.json"

    route_map_image = args.map_image
    if args.plan_on_mirror_x:
        route_map_image = args.output_dir / "planning" / "autonomous_map_visual_mirror_x.png"
        route_map_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(route_map_image), cv2.flip(mirror.load_map_image(args.map_image), 1))

    polylines = mirror.route_polylines(
        args.route_config,
        route_map_image,
        args.clearance_px,
        planned=True,
        scope="all",
    )
    route_transform = "current_code_transform_no_map_offset"
    if args.plan_on_mirror_x:
        route_transform = "current_code_transform_planned_on_mirror_x_visual"
    if args.route_mirror_x:
        polylines = [
            [(mirror.MAP_TEXTURE_SIZE - px, py) for px, py in polyline]
            for polyline in polylines
        ]
        route_transform = "route_points_mirror_x"
    triangle_count = mirror.write_route_preview_dae(
        route_mesh_path,
        polylines,
        offset_x=0.0,
        offset_y=0.0,
        z=0.045,
        thickness_m=args.thickness_m,
    )
    write_world(world_path, args.map_mesh, route_mesh_path)

    info = {
        "world": str(world_path),
        "map_mesh": str(args.map_mesh),
        "route_mesh": str(route_mesh_path),
        "route_scope": "all",
        "route_planned": True,
        "map_visual_mirror_axis": "x",
        "map_pose": [0.0, 0.3, 0.0, 0.0, 0.0, 0.0],
        "route_overlay_transform": route_transform,
        "route_polylines": len(polylines),
        "route_triangles": triangle_count,
    }
    info_path.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"Generated mirror overlay view world: {world_path}")
    print(f"Generated route overlay mesh: {route_mesh_path}")
    print(f"Info: {info_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
