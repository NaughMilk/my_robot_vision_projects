# My Robot Vision - Signless Map Bundle

Bundle nay chua mot ROS 2 workspace toi gian cho map hien tai:

- Map Gazebo: `src/my_robot_vision/worlds/autonomous_map_3d.world`
- Route graph mac dinh: `src/my_robot_vision/config/autonomous_route_graph.json`
- Map image cho route mask: `src/my_robot_vision/maps/autonomous_map_visual.png`
- Logic dieu khien/route/vision: `src/my_robot_vision/my_robot_vision/`
- Models can thiet: map mesh/texture, xe `track_preview_car`, va texture template cho nhan dien bien bao

Bundle khong chua cac map/world khac, backup, build/install/log, artifacts, hay route preview mau tim. World trong bundle chi co map nen va xe, khong co model bien bao hoac den giao thong.

## Yeu cau

May dich can co ROS 2 Foxy va Gazebo Classic voi cac package:

```bash
sudo apt update
sudo apt install -y \
  ros-foxy-desktop \
  ros-foxy-gazebo-ros-pkgs \
  ros-foxy-gazebo-plugins \
  ros-foxy-cv-bridge \
  python3-colcon-common-extensions \
  python3-numpy \
  python3-opencv
```

Neu may da co ROS 2 Foxy/Gazebo san roi thi co the bo qua buoc apt.

## Build

```bash
cd my_robot_vision_map_signless_bundle_20260610
source /opt/ros/foxy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

## Chay map nay

Launch da dat mac dinh dung world/route/map image trong bundle, va da tat sign/traffic light vi world nay khong co bien/den.

```bash
ros2 launch my_robot_vision gazebo_autonomous_map.launch.py \
  gui:=true \
  camera_viewer:=true \
  reverse_direction:=true \
  route_explore_enabled:=false \
  branch_probability:=0.0 \
  route_offset_y_m:=0.3 \
  parking_auto_route_enabled:=false \
  route_rejoin_enabled:=false \
  initial_pose_reset_enabled:=false
```

Neu muon chay headless:

```bash
ros2 launch my_robot_vision gazebo_autonomous_map.launch.py \
  gui:=false \
  camera_viewer:=false \
  reverse_direction:=true \
  route_explore_enabled:=false \
  branch_probability:=0.0 \
  route_offset_y_m:=0.3 \
  parking_auto_route_enabled:=false \
  route_rejoin_enabled:=false \
  initial_pose_reset_enabled:=false
```

## Ghi chu ve logic

- Route graph da duoc copy vao `config/autonomous_route_graph.json`, nen khong can truyen `route_config:=...`.
- Map image da duoc copy vao `maps/autonomous_map_visual.png`, nen khong can truyen `map_image:=...`.
- World da duoc sua portable URI thanh `model://autonomous_map_3d_mirror_x/...`, nen khong phu thuoc duong dan duong dan may goc.
- Sign/traffic-light logic van co trong source, nhung launch mac dinh tat cac module do cho map signless nay.

## Kiem tra nhanh noi dung

World hop le khi chi thay hai model chinh:

- `autonomous_map_3d_mirror_x`
- `track_preview_car`

Khong nen co `palette_*`, `traffic_light_*`, `road_sign_*` trong world file.
