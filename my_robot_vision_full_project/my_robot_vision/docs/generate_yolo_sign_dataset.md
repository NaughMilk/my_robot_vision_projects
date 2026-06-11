# Generate YOLO road-sign dataset

This project can generate a synthetic YOLO dataset from the existing Gazebo sign
textures. It is meant for training a detector on many camera-like views of the
same sign model without manually taking screenshots.

## What the generator does

The current 3D road-sign models are flat sign plates with a texture, mounted on
a pole. The generator mimics that camera view by:

- Loading the sign texture from `models/road_sign_*/materials/textures`
- Placing it on a square sign plate like the Gazebo model
- Projecting the plate with random yaw, pitch, roll, scale, and focal length
- Adding a pole, shadow, road/grass/sidewalk backgrounds, blur, noise, and
  brightness changes
- Writing YOLO labels for the sign face only

The pole is visible in the image but is not part of the YOLO bounding box.

## Output layout

```text
datasets/sign_yolo_synthetic/
  data.yaml
  classes.txt
  generation_config.json
  preview.jpg
  images/train/*.jpg
  images/val/*.jpg
  images/test/*.jpg
  labels/train/*.txt
  labels/val/*.txt
  labels/test/*.txt
```

Each label file uses YOLO format:

```text
class_id center_x center_y width height
```

All coordinates are normalized from `0.0` to `1.0`.

## Generate one sign class

Example for right-turn sign:

```bash
python3 tools/generate_yolo_sign_dataset.py \
  --output-dir datasets/sign_yolo_turn_right \
  --sign turn_right \
  --samples-per-class 300 \
  --negative-samples 60 \
  --overwrite
```

Supported sign keys:

- `go_straight`
- `turn_left`
- `turn_right`
- `parking`
- `speed_limit`

## Generate all sign classes

```bash
python3 tools/generate_yolo_sign_dataset.py \
  --output-dir datasets/sign_yolo_all \
  --all-signs \
  --samples-per-class 300 \
  --negative-samples 200 \
  --overwrite
```

## Train YOLO

With Ultralytics YOLO installed:

```bash
yolo detect train \
  data=datasets/sign_yolo_turn_right/data.yaml \
  model=yolov8n.pt \
  imgsz=640 \
  epochs=80
```

If training one sign only, the dataset has one class. If training all signs,
class ids follow `classes.txt`.

## Notes

- This is faster than Gazebo screenshots and creates perfect labels.
- It is still synthetic, so real/simulator camera frames should be mixed in
  later if possible.
- For best results, generate many examples with different seeds and combine
  them.
