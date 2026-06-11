# my_robot_vision

ROS2 runtime demo cho bai tap robot bam line. Package nay mo phong day du vong lap:

```text
/robot_pose -> virtual_camera -> /camera/image_raw -> line_follower -> /cmd_vel -> robot_sim
```

## Node

- `virtual_camera`: tao camera ao va publish anh BGR len `/camera/image_raw`.
- `line_follower`: nhan anh, loc vang bang HSV, bam tam cum vach vang, publish `/cmd_vel`.
- `robot_sim`: nhan `/cmd_vel`, cap nhat dong hoc 2D, publish `/robot_pose`, hien thi robot tren sa ban.

## Imported map

- Map goc tu CorelDRAW duoc luu trong `assets/autonomous_map_in_corel.cdr`.
- Preview map duoc import vao mo phong o `maps/autonomous_map_visual.png`.
- Khi package tim thay map nay, `world.py` se dung map imported lam nen sa ban va ve guide loop mau vang ben tren de line follower van bam duoc.

## Gazebo 3D map

Package cung co ban 3D xuat tu map Corel:

- `models/autonomous_map_3d/meshes/autonomous_map_3d.dae`: mesh Collada co texture.
- `models/autonomous_map_3d/materials/textures/autonomous_map_visual.png`: texture map.
- `worlds/autonomous_map_3d.world`: Gazebo world nap model 3D.
- `launch/gazebo_autonomous_map.launch.py`: launch Gazebo voi world 3D.

Cai Gazebo cho ROS2 Foxy:

```bash
sudo apt install -y ros-foxy-gazebo-ros-pkgs gazebo11
```

Chay map 3D:

```bash
ros2 launch my_robot_vision gazebo_autonomous_map.launch.py
```

Neu doi map 2D va muon sinh lai mesh 3D:

```bash
cd ~/ros2_ws/src/my_robot_vision
python3 tools/generate_autonomous_map_3d.py
```

## Cai dependency

Tren Ubuntu/WSL2 da cai ROS2:

```bash
sudo apt update
sudo apt install ros-$ROS_DISTRO-cv-bridge python3-opencv python3-numpy
```

## Build

```bash
cd ~/ros2_ws
colcon build --packages-select my_robot_vision
source install/setup.bash
```

## Chay demo mot lenh

```bash
ros2 launch my_robot_vision line_follow_demo.launch.py
```

Neu may khong mo duoc cua so OpenCV:

```bash
ros2 launch my_robot_vision line_follow_demo.launch.py show_debug:=false show_gui:=false
```

Khi tat GUI, van co the xem anh bang:

```bash
ros2 run rqt_image_view rqt_image_view
```

Chon topic:

- `/camera/image_raw`
- `/line_follower/debug_image`
- `/robot_sim/world_image`

## Chay tung node rieng

Mo 3 terminal, terminal nao cung can:

```bash
cd ~/ros2_ws
source install/setup.bash
```

Terminal 1:

```bash
ros2 run my_robot_vision robot_sim
```

Terminal 2:

```bash
ros2 run my_robot_vision virtual_camera
```

Terminal 3:

```bash
ros2 run my_robot_vision line_follower
```

## Lenh kiem tra

```bash
ros2 topic list
ros2 topic echo /cmd_vel
ros2 topic hz /camera/image_raw
```

## Noi dung de thuyet trinh

Thuat toan nhan dien vach vang trong khong gian mau HSV, chi lay ROI o phan duoi anh. Sau do lay tam cua cum vach vang bang 3 scan line, uu tien cum gan du doan truoc do de tranh nhay sang doan line khac o cua 90 do, roi dung dieu khien PD de sinh `linear.x` va `angular.z`. Neu mat line ngan han, robot tiep tuc di bang bo nho loi cu; neu mat qua lau thi quay ve trang thai `SEARCH`.
