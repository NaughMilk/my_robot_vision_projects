# Route Drawing Lab

Tool nay cho ban chinh route bang tay ngay trong Gazebo:

- World nhe chi co map mirror-X.
- Script spawn cac waypoint mau vang.
- Ban keo waypoint trong Gazebo bang translate tool.
- Script noi cac waypoint theo thu tu bang line mau hong.
- `save` se luu JSON/CSV/PNG de Codex phan tich va chuyen ve route config.

No khong sua `config/autonomous_route_graph.json`.

## Chay

```bash
cd /home/quocbao/my_robot_vision_full_project/my_robot_vision
source /opt/ros/foxy/setup.bash
source /home/quocbao/my_robot_vision_full_project/install/setup.bash
python3 route_drawing_lab/run_gazebo_route_drawer.py --template middle
```

Neu Gazebo dang bi ket tu lan truoc:

```bash
pkill -KILL -x gazebo
pkill -KILL -x gzserver
pkill -KILL -x gzclient
```

## Cach dung trong Gazebo

1. Bam chon waypoint mau vang.
2. Bam tool translate tren toolbar.
3. Keo waypoint vao giua lane.
4. Line hong se noi waypoint theo thu tu index.
5. Quay lai terminal, go:

```text
save
```

Hoac thoat va save:

```text
quit
```

## Lenh terminal

```text
help                  hien help
print                 in toa do waypoint
save [label]          luu JSON/CSV/PNG
add X Y               them waypoint bang toa do Gazebo model meters
addpx PX PY           them waypoint bang toa do pixel route/code
insert I X Y          chen waypoint o index I
insertpx I PX PY      chen waypoint bang pixel route/code
delete I              xoa waypoint index I
refresh               ve lai line noi waypoint
quit                  save va thoat
```

## Template

```bash
python3 route_drawing_lab/run_gazebo_route_drawer.py --template middle
python3 route_drawing_lab/run_gazebo_route_drawer.py --template top_cut
python3 route_drawing_lab/run_gazebo_route_drawer.py --template lower_cut
python3 route_drawing_lab/run_gazebo_route_drawer.py --template empty
```

## Output

Moi lan chay se tao folder:

```text
route_drawing_lab/outputs/session_YYYYMMDD_HHMMSS/
```

File quan trong:

- `drawn_route.json`: toa do model + pixel route + pixel tren visual mirror.
- `drawn_route.csv`: bang toa do de xem nhanh.
- `drawn_route_overlay.png`: anh 2D de check nhanh.
- `route_drawing_world.world`: world dung de ve.

Gui `drawn_route.json` hoac ca folder output cho Codex de chuyen thanh route config.
