# Route Recording Lab

Folder nay dung de tao waypoint/road graph rieng tu Gazebo 3D. No khong sua code tu lai hien tai.

## Muc tieu

- Tao world moi khong co bien bao tu world goc.
- Spawn xe o mot goc ngoai.
- Ban lai tay xe bang ban phim.
- Script tu ghi pose xe tu `/model_states`.
- Cac lan di nguoc chieu/cung duong se duoc merge thanh canh 2 chieu trong graph.
- Output luu trong `route_recording_lab/outputs/<session>/`.

## Chay

```bash
cd /home/quocbao/my_robot_vision_full_project/my_robot_vision
source /opt/ros/foxy/setup.bash
source /home/quocbao/my_robot_vision_full_project/install/setup.bash
python3 route_recording_lab/run_recording_session.py
```

Mac dinh script se dung `/home/quocbao/Downloads/map_1` neu file nay ton tai. Neu muon chi dinh world khac:

```bash
python3 route_recording_lab/run_recording_session.py --source-world /home/quocbao/Downloads/map_1
```

Neu chi muon tao world khong co bien ma chua lai:

```bash
python3 route_recording_lab/run_recording_session.py --generate-world-only
```

Neu muon chay Gazebo khong GUI:

```bash
python3 route_recording_lab/run_recording_session.py --headless
```

## Phim dieu khien

- `w` tang toc tien
- `s` giam toc / lui
- `a` tang quay trai
- `d` tang quay phai
- `e` dua toc do quay ve 0
- `space` dung xe
- `r` pause/resume recording
- `n` tao segment moi, dung sau khi reset hoac sau mot doan lai sai
- `p` save ngay
- `1` reset top-left
- `2` reset top-right
- `3` reset bottom-right
- `4` reset bottom-left
- `q` save va thoat

## File output

- `recorded_route_graph.json`: du lieu chinh, gom raw samples + graph node/edge.
- `recorded_route_raw.csv`: pose xe theo thoi gian.
- `recorded_route_graph_nodes.csv`: node da merge.
- `recorded_route_graph_edges.csv`: edge 2 chieu da merge.
- `recorded_route_overlay.png`: anh kiem tra nhanh tren map 2D.
- `signless_world_info.json`: thong tin world signless da tao.
- `gazebo_session.log`: log Gazebo cua session.

## Ghi chu quan trong

Day la tool calibration/offline. Khi demo cham diem thi khong dung teleop. Sau khi record du graph, robot tu lai van phai tu nhan bien/den va tu quyet dinh.
