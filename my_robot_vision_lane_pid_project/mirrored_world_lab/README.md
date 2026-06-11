# Mirrored World Lab

Folder nay dung de test gia thuyet: world Gazebo 3D dang bi mirror so voi map/route 2D.

No khong sua world goc, khong sua node tu lai hien tai. Script chi tao world test rieng trong `mirrored_world_lab/generated/`.

## Tao world mirror

```bash
cd /home/quocbao/my_robot_vision_full_project/my_robot_vision
python3 mirrored_world_lab/generate_mirrored_world.py
```

Mac dinh script tao:

```text
mirrored_world_lab/generated/autonomous_map_3d_mirror_x_route_preview.world
```

Trong world nay:

- Map visual duoc mirror trai-phai theo truc X.
- Cac model top-level nhu xe, den, bien duoc mirror pose theo cung truc.
- Saved `<state>` bi xoa de Gazebo khong load lai pose cu.
- Co them route preview tren mat duong.

## Mo Gazebo de xem

```bash
cd /home/quocbao/my_robot_vision_full_project
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch gazebo_ros gazebo.launch.py world:=/home/quocbao/my_robot_vision_full_project/my_robot_vision/mirrored_world_lab/generated/autonomous_map_3d_mirror_x_route_preview.world gui:=true
```

## Y nghia route preview

- Mau hong/magenta: route theo transform code hien tai.
- Mau xanh/cyan: route theo transform co cong pose map visual `(0, 0.3)`.

Neu mau nao nam dung giua duong hon trong Gazebo, ta biet transform nao phai dung cho planner/controller.

## Tao ban mirror truc khac

Neu mirror X chua dung, tao thu mirror Y:

```bash
python3 mirrored_world_lab/generate_mirrored_world.py \
  --axis y \
  --output-world mirrored_world_lab/generated/autonomous_map_3d_mirror_y_route_preview.world
```

Mo:

```bash
ros2 launch gazebo_ros gazebo.launch.py world:=/home/quocbao/my_robot_vision_full_project/my_robot_vision/mirrored_world_lab/generated/autonomous_map_3d_mirror_y_route_preview.world gui:=true
```

## Chay auto node tren world mirror

Sau khi nhin route preview thay khop, co the chay node hien tai tren world mirror:

```bash
cd /home/quocbao/my_robot_vision_full_project
source /opt/ros/foxy/setup.bash
source install/setup.bash
ros2 launch my_robot_vision gazebo_autonomous_map.launch.py world:=/home/quocbao/my_robot_vision_full_project/my_robot_vision/mirrored_world_lab/generated/autonomous_map_3d_mirror_x_route_preview.world gui:=true camera_viewer:=true
```

Day van chi la test world rieng, khong anh huong world goc.
