# Build 3D road signs from 2D map data

This note documents the sign-building logic used in this project, in a form
that can be copied to another Gazebo/ROS project.

## Core idea

The 2D map is a square texture. In this project:

- Texture size: `256 x 256 px`
- Gazebo map size: `6.0 x 6.0 m`
- Map visual offset: `(x=0.0 m, y=0.3 m)`
- Image origin: top-left
- Gazebo map origin: center of map

Conversion from pixel to Gazebo:

```python
model_x = (px / 256.0 - 0.5) * 6.0 + offset_x
model_y = (0.5 - py / 256.0) * 6.0 + offset_y
```

The `y` axis is flipped because image `y` grows downward, while Gazebo `y`
grows upward in the map plane.

## Sign orientation

The generated sign models put the visible sign face at local `y = -0.011`.
That means the face normal points along local `-Y`. If `model_yaw = 0`, the
sign face looks toward Gazebo `-Y`.

Therefore:

```python
face_yaw = model_yaw - pi / 2
model_yaw = face_yaw + pi / 2
```

In most world files we directly use `model_yaw`:

- `0 deg` -> model yaw `0`
- `90 deg` -> model yaw `1.570796`
- `180 deg` -> model yaw `3.141593`
- `270 deg` -> model yaw `-1.570796`

## Model naming

Detection logic recognizes signs from model names. Keep names descriptive:

- `palette_turn_right_03_yaw_270`
- `palette_turn_left_01_yaw_090`
- `palette_go_straight_00_yaw_000`
- `palette_parking_00_yaw_000`
- `palette_speed_limit_00_yaw_000`
- `palette_stop_00_yaw_000`

The important tokens are `turn_right`, `turn_left`, `go_straight`, `parking`,
`speed_limit`, and `stop`.

## 3D model structure

For textured signs such as right/left/parking/speed:

- A small circular base cylinder
- A thin vertical pole
- A white back plate box
- A thin front plate box with Gazebo material texture

The model is static and included in the world with:

```xml
<include>
  <name>palette_turn_right_03_yaw_270</name>
  <uri>model://road_sign_turn_right</uri>
  <pose>x y 0 0 0 yaw</pose>
</include>
```

## Standalone script

Full reusable code is in:

```text
tools/build_3d_signs_from_2d.py
```

It can:

1. Generate road sign model folders from 2D sign textures.
2. Convert pixel positions to Gazebo coordinates.
3. Insert sign `<include>` blocks into a Gazebo world.

## Example signs JSON

```json
[
  {
    "name": "palette_turn_right_03_yaw_270",
    "kind": "turn_right",
    "px": 43.0,
    "py": 126.0,
    "yaw_deg": 270
  },
  {
    "name": "palette_parking_00_yaw_000",
    "kind": "parking",
    "px": 156.0,
    "py": 103.0,
    "yaw_deg": 0
  }
]
```

Supported `kind` values:

- `go_straight`
- `turn_left`
- `turn_right`
- `parking`
- `speed_limit`
- `stop`
- `arrow`

## Example usage

First generate example JSON:

```bash
python3 tools/build_3d_signs_from_2d.py \
  --write-example-signs /tmp/signs.json
```

Then generate/patched world:

```bash
python3 tools/build_3d_signs_from_2d.py \
  --input-world worlds/autonomous_map_3d.world \
  --output-world worlds/autonomous_map_3d_with_signs.world \
  --signs-json /tmp/signs.json \
  --models-dir models \
  --texture-dir sign_textures \
  --create-models
```

`sign_textures` should contain these files if textured signs are used:

- `go_straight.jpg`
- `turn_left.png`
- `turn_right.png`
- `parking.png`
- `limit_speed.jpg`

`stop` and `arrow` can be generated procedurally and do not require textures.

## Important practical notes

- Add the generated `models` folder to `GAZEBO_MODEL_PATH`.
- Keep world URIs as `model://...`, not `file:///home/...`, if the world must
  run on another machine.
- If you move a sign in Gazebo GUI, save the world and keep the same model
  name so detection still knows the sign type.
- The route/controller uses the sign model name to attach the camera detection
  to a real scene sign. A camera crop alone is not enough for reliable routing.
