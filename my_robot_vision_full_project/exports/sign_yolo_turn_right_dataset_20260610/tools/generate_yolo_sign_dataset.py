#!/usr/bin/env python3
"""Generate a synthetic YOLO dataset for road-sign detection.

The Gazebo sign models in this project are mostly flat textured plates mounted
on a pole. For YOLO training, a lightweight synthetic renderer is usually much
faster than launching Gazebo hundreds of times: it projects the sign texture as
if a camera viewed the 3D plate from many yaw/pitch/roll angles, then writes
YOLO images and labels.

The output layout is:

  output_dir/
    data.yaml
    classes.txt
    generation_config.json
    preview.jpg
    images/{train,val,test}/*.jpg
    labels/{train,val,test}/*.txt
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class SignSpec:
    key: str
    yolo_name: str
    texture_path: Path


DEFAULT_SIGNS: dict[str, SignSpec] = {
    "go_straight": SignSpec(
        key="go_straight",
        yolo_name="go_straight_sign",
        texture_path=PROJECT_ROOT / "models/road_sign_go_straight/materials/textures/go_straight.jpg",
    ),
    "turn_left": SignSpec(
        key="turn_left",
        yolo_name="turn_left_sign",
        texture_path=PROJECT_ROOT / "models/road_sign_turn_left/materials/textures/turn_left.png",
    ),
    "turn_right": SignSpec(
        key="turn_right",
        yolo_name="turn_right_sign",
        texture_path=PROJECT_ROOT / "models/road_sign_turn_right/materials/textures/turn_right.png",
    ),
    "parking": SignSpec(
        key="parking",
        yolo_name="parking_sign",
        texture_path=PROJECT_ROOT / "models/road_sign_parking/materials/textures/parking.png",
    ),
    "speed_limit": SignSpec(
        key="speed_limit",
        yolo_name="speed_limit_sign",
        texture_path=PROJECT_ROOT / "models/road_sign_speed_limit/materials/textures/limit_speed.jpg",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "datasets/sign_yolo_synthetic",
        help="Dataset output directory.",
    )
    parser.add_argument(
        "--sign",
        action="append",
        choices=sorted(DEFAULT_SIGNS),
        help="Sign kind to generate. Can be repeated. Defaults to turn_right.",
    )
    parser.add_argument(
        "--all-signs",
        action="store_true",
        help="Generate every built-in textured sign class.",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=300,
        help="Number of positive images per selected class.",
    )
    parser.add_argument("--negative-samples", type=int, default=0, help="Images without signs.")
    parser.add_argument("--image-width", type=int, default=640)
    parser.add_argument("--image-height", type=int, default=480)
    parser.add_argument("--train-ratio", type=float, default=0.80)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--format", choices=("jpg", "png"), default="jpg")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Clear an existing output directory before generating.",
    )
    parser.add_argument(
        "--min-object-px",
        type=int,
        default=28,
        help="Minimum bbox width/height accepted for a generated object.",
    )
    return parser.parse_args()


def ensure_clean_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists() and overwrite:
        for child in path.iterdir():
            if child.is_dir():
                for nested in sorted(child.rglob("*"), reverse=True):
                    if nested.is_file() or nested.is_symlink():
                        nested.unlink()
                    elif nested.is_dir():
                        nested.rmdir()
                child.rmdir()
            else:
                child.unlink()
    path.mkdir(parents=True, exist_ok=True)


def load_sign_texture(path: Path, canvas_size: int = 512) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(f"Could not read sign texture: {path}")

    if image.ndim == 2:
        bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        alpha = np.full(image.shape, 255, dtype=np.uint8)
    elif image.shape[2] == 3:
        bgr = image
        alpha = np.full(image.shape[:2], 255, dtype=np.uint8)
    else:
        bgr = image[:, :, :3]
        alpha = image[:, :, 3]

    rgba = np.dstack([bgr, alpha])
    rgba = crop_transparent_border(rgba)

    # Gazebo maps these textures to a square sign plate. We mimic that by
    # compositing the texture on a square white plate.
    plate = np.zeros((canvas_size, canvas_size, 4), dtype=np.uint8)
    plate[:, :, :3] = 245
    plate[:, :, 3] = 255

    resized = cv2.resize(rgba, (canvas_size, canvas_size), interpolation=cv2.INTER_AREA)
    alpha_f = resized[:, :, 3:4].astype(np.float32) / 255.0
    plate[:, :, :3] = (
        resized[:, :, :3].astype(np.float32) * alpha_f
        + plate[:, :, :3].astype(np.float32) * (1.0 - alpha_f)
    ).astype(np.uint8)

    # A small edge helps YOLO see the plate under bright backgrounds.
    cv2.rectangle(plate, (0, 0), (canvas_size - 1, canvas_size - 1), (238, 238, 238, 255), 8)
    cv2.rectangle(plate, (6, 6), (canvas_size - 7, canvas_size - 7), (255, 255, 255, 255), 3)
    return plate


def crop_transparent_border(rgba: np.ndarray) -> np.ndarray:
    alpha = rgba[:, :, 3]
    coords = cv2.findNonZero((alpha > 3).astype(np.uint8))
    if coords is None:
        return rgba
    x, y, w, h = cv2.boundingRect(coords)
    pad = max(2, int(max(w, h) * 0.02))
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(rgba.shape[1], x + w + pad)
    y1 = min(rgba.shape[0], y + h + pad)
    return rgba[y0:y1, x0:x1]


def rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cr, sr = math.cos(roll), math.sin(roll)

    ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    rz = np.array([[cr, -sr, 0.0], [sr, cr, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)
    return rz @ rx @ ry


def project_plate_quad(
    plate_size_px: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
    focal_px: float,
) -> np.ndarray:
    half = plate_size_px * 0.5
    corners = np.array(
        [
            [-half, -half, 0.0],
            [half, -half, 0.0],
            [half, half, 0.0],
            [-half, half, 0.0],
        ],
        dtype=np.float32,
    )
    rotation = rotation_matrix(
        math.radians(yaw_deg),
        math.radians(pitch_deg),
        math.radians(roll_deg),
    )
    rotated = corners @ rotation.T
    distance = focal_px
    z = np.maximum(distance + rotated[:, 2], 1.0)
    projected = np.column_stack(
        [
            focal_px * rotated[:, 0] / z,
            focal_px * rotated[:, 1] / z,
        ]
    ).astype(np.float32)
    return projected


def make_background(width: int, height: int, rng: random.Random) -> np.ndarray:
    mode = rng.choice(("road", "road_grass", "sidewalk", "soft_noise"))
    if mode == "soft_noise":
        base = np.zeros((height, width, 3), dtype=np.uint8)
        color = np.array(
            [rng.randint(90, 150), rng.randint(95, 155), rng.randint(95, 165)],
            dtype=np.uint8,
        )
        base[:, :] = color
    else:
        base = np.zeros((height, width, 3), dtype=np.uint8)
        road_gray = rng.randint(55, 95)
        base[:, :] = (road_gray, road_gray, road_gray + rng.randint(0, 8))

        if mode in {"road_grass", "sidewalk"}:
            horizon = rng.randint(int(height * 0.45), int(height * 0.72))
            if mode == "road_grass":
                base[:horizon, :] = (
                    rng.randint(35, 80),
                    rng.randint(115, 170),
                    rng.randint(35, 80),
                )
            else:
                concrete = rng.randint(145, 190)
                base[:horizon, :] = (concrete, concrete, concrete)

        for _ in range(rng.randint(1, 5)):
            x0 = rng.randint(-width // 3, width)
            y0 = rng.randint(0, height)
            x1 = x0 + rng.randint(width // 8, width // 2)
            y1 = y0 + rng.randint(4, 16)
            color = rng.choice([(220, 220, 220), (195, 185, 65), (120, 120, 120)])
            cv2.line(base, (x0, y0), (x1, y1), color, rng.randint(2, 7), cv2.LINE_AA)

        if rng.random() < 0.55:
            stripe_y = rng.randint(int(height * 0.55), height - 20)
            stripe_w = rng.randint(24, 42)
            for x in range(rng.randint(-60, 20), width, rng.randint(48, 78)):
                cv2.rectangle(
                    base,
                    (x, stripe_y),
                    (x + stripe_w, stripe_y + rng.randint(6, 14)),
                    (210, 210, 210),
                    -1,
                )

    noise = np.random.normal(0.0, rng.uniform(4.0, 12.0), (height, width, 3)).astype(np.float32)
    out = np.clip(base.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.35:
        k = rng.choice((3, 5))
        out = cv2.GaussianBlur(out, (k, k), 0)
    return out


def jitter_rgba(rgba: np.ndarray, rng: random.Random) -> np.ndarray:
    out = rgba.copy()
    bgr = out[:, :, :3].astype(np.float32)
    contrast = rng.uniform(0.75, 1.30)
    brightness = rng.uniform(-28.0, 30.0)
    bgr = np.clip((bgr - 127.5) * contrast + 127.5 + brightness, 0, 255)

    hsv = cv2.cvtColor(bgr.astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(0.75, 1.35), 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(0.80, 1.25), 0, 255)
    out[:, :, :3] = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    if rng.random() < 0.45:
        k = rng.choice((3, 5))
        out[:, :, :3] = cv2.GaussianBlur(out[:, :, :3], (k, k), 0)
    return out


def alpha_blend_rgba(background: np.ndarray, foreground_rgba: np.ndarray) -> np.ndarray:
    alpha = foreground_rgba[:, :, 3:4].astype(np.float32) / 255.0
    blended = (
        foreground_rgba[:, :, :3].astype(np.float32) * alpha
        + background.astype(np.float32) * (1.0 - alpha)
    )
    return np.clip(blended, 0, 255).astype(np.uint8)


def clamp_bbox(x0: float, y0: float, x1: float, y1: float, width: int, height: int) -> tuple[int, int, int, int]:
    return (
        int(max(0, min(width - 1, math.floor(x0)))),
        int(max(0, min(height - 1, math.floor(y0)))),
        int(max(0, min(width - 1, math.ceil(x1)))),
        int(max(0, min(height - 1, math.ceil(y1)))),
    )


def yolo_line(class_id: int, bbox: tuple[int, int, int, int], width: int, height: int) -> str:
    x0, y0, x1, y1 = bbox
    cx = ((x0 + x1) * 0.5) / width
    cy = ((y0 + y1) * 0.5) / height
    bw = max(1.0, x1 - x0) / width
    bh = max(1.0, y1 - y0) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}\n"


def draw_pole_and_shadow(image: np.ndarray, quad: np.ndarray, rng: random.Random) -> None:
    shadow = np.zeros_like(image)
    shadow_offset = np.array([rng.randint(5, 24), rng.randint(8, 28)], dtype=np.float32)
    shadow_quad = (quad + shadow_offset).astype(np.int32)
    cv2.fillConvexPoly(shadow, shadow_quad, (0, 0, 0), cv2.LINE_AA)
    image[:] = cv2.addWeighted(image, 1.0, shadow, rng.uniform(0.10, 0.24), 0)

    bottom_center = np.mean([quad[2], quad[3]], axis=0)
    pole_len = rng.uniform(75, 190)
    pole_skew = rng.uniform(-20, 22)
    start = tuple(np.round(bottom_center).astype(int))
    end = tuple(np.round(bottom_center + np.array([pole_skew, pole_len])).astype(int))
    color = rng.choice([(170, 170, 165), (190, 190, 184), (135, 135, 130)])
    cv2.line(image, start, end, color, rng.randint(5, 9), cv2.LINE_AA)
    cv2.circle(image, end, rng.randint(10, 18), tuple(max(0, c - 30) for c in color), -1, cv2.LINE_AA)


def render_one(
    texture_rgba: np.ndarray,
    width: int,
    height: int,
    rng: random.Random,
    min_object_px: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    source_h, source_w = texture_rgba.shape[:2]
    src_quad = np.array(
        [[0, 0], [source_w - 1, 0], [source_w - 1, source_h - 1], [0, source_h - 1]],
        dtype=np.float32,
    )

    for _ in range(200):
        background = make_background(width, height, rng)
        plate_size = rng.uniform(width * 0.16, width * 0.42)
        focal = rng.uniform(360.0, 780.0)
        yaw = rng.uniform(-68.0, 68.0)
        pitch = rng.uniform(-38.0, 34.0)
        roll = rng.uniform(-18.0, 18.0)
        quad = project_plate_quad(plate_size, yaw, pitch, roll, focal)

        min_xy = quad.min(axis=0)
        max_xy = quad.max(axis=0)
        object_w = max_xy[0] - min_xy[0]
        object_h = max_xy[1] - min_xy[1]
        if object_w < min_object_px or object_h < min_object_px:
            continue

        margin = int(max(10, min(width, height) * 0.04))
        center_min_x = margin - min_xy[0]
        center_max_x = width - margin - max_xy[0]
        center_min_y = margin - min_xy[1]
        center_max_y = height - margin - max_xy[1]
        if center_min_x >= center_max_x or center_min_y >= center_max_y:
            continue

        center = np.array(
            [
                rng.uniform(center_min_x, center_max_x),
                rng.uniform(center_min_y, center_max_y),
            ],
            dtype=np.float32,
        )
        dst_quad = quad + center

        draw_pole_and_shadow(background, dst_quad, rng)
        sign_rgba = jitter_rgba(texture_rgba, rng)
        matrix = cv2.getPerspectiveTransform(src_quad, dst_quad.astype(np.float32))
        warped = cv2.warpPerspective(
            sign_rgba,
            matrix,
            (width, height),
            flags=cv2.INTER_AREA,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0),
        )

        if rng.random() < 0.25:
            add_occlusion(warped, rng)

        image = alpha_blend_rgba(background, warped)
        image = finish_camera_effects(image, rng)

        alpha_mask = warped[:, :, 3] > 12
        coords = cv2.findNonZero(alpha_mask.astype(np.uint8))
        if coords is None:
            continue
        x, y, w, h = cv2.boundingRect(coords)
        bbox = clamp_bbox(x, y, x + w, y + h, width, height)
        if bbox[2] - bbox[0] < min_object_px or bbox[3] - bbox[1] < min_object_px:
            continue
        return image, bbox

    raise RuntimeError("Could not render a valid sign after many attempts")


def add_occlusion(warped_rgba: np.ndarray, rng: random.Random) -> None:
    alpha = warped_rgba[:, :, 3]
    coords = cv2.findNonZero((alpha > 12).astype(np.uint8))
    if coords is None:
        return
    x, y, w, h = cv2.boundingRect(coords)
    occ_w = rng.randint(max(4, w // 12), max(5, w // 4))
    occ_h = rng.randint(max(4, h // 12), max(5, h // 4))
    ox = rng.randint(x, max(x, x + w - occ_w))
    oy = rng.randint(y, max(y, y + h - occ_h))
    warped_rgba[oy : oy + occ_h, ox : ox + occ_w, 3] = 0


def finish_camera_effects(image: np.ndarray, rng: random.Random) -> np.ndarray:
    out = image
    if rng.random() < 0.35:
        noise = np.random.normal(0.0, rng.uniform(2.0, 8.0), out.shape).astype(np.float32)
        out = np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    if rng.random() < 0.16:
        kernel_size = rng.choice((3, 5, 7))
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.float32)
        if rng.random() < 0.5:
            kernel[kernel_size // 2, :] = 1.0
        else:
            kernel[:, kernel_size // 2] = 1.0
        kernel /= kernel_size
        out = cv2.filter2D(out, -1, kernel)
    return out


def split_name(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    if index < train_end:
        return "train"
    if index < val_end:
        return "val"
    return "test"


def write_dataset_metadata(
    output_dir: Path,
    selected_signs: list[SignSpec],
    args: argparse.Namespace,
) -> None:
    names = [spec.yolo_name for spec in selected_signs]
    (output_dir / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    yaml_lines = [
        f"path: {output_dir}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    yaml_lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    (output_dir / "data.yaml").write_text("\n".join(yaml_lines) + "\n", encoding="utf-8")

    config = {
        "selected_signs": [spec.key for spec in selected_signs],
        "class_names": names,
        "samples_per_class": args.samples_per_class,
        "negative_samples": args.negative_samples,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "train_ratio": args.train_ratio,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "format": args.format,
    }
    (output_dir / "generation_config.json").write_text(
        json.dumps(config, indent=2) + "\n",
        encoding="utf-8",
    )


def write_preview(output_dir: Path, image_paths: list[Path], width: int = 4) -> None:
    if not image_paths:
        return
    chosen = image_paths[: min(16, len(image_paths))]
    thumbs = []
    thumb_w, thumb_h = 180, 135
    for path in chosen:
        image = cv2.imread(str(path))
        if image is None:
            continue
        thumbs.append(cv2.resize(image, (thumb_w, thumb_h), interpolation=cv2.INTER_AREA))
    if not thumbs:
        return
    rows = int(math.ceil(len(thumbs) / width))
    canvas = np.full((rows * thumb_h, width * thumb_w, 3), 245, dtype=np.uint8)
    for idx, thumb in enumerate(thumbs):
        row, col = divmod(idx, width)
        canvas[row * thumb_h : (row + 1) * thumb_h, col * thumb_w : (col + 1) * thumb_w] = thumb
    cv2.imwrite(str(output_dir / "preview.jpg"), canvas)


def generate_negative_image(width: int, height: int, rng: random.Random) -> np.ndarray:
    image = make_background(width, height, rng)
    if rng.random() < 0.4:
        for _ in range(rng.randint(1, 4)):
            x0 = rng.randint(0, max(1, width - 80))
            y0 = rng.randint(0, max(1, height - 80))
            x1 = min(width - 1, x0 + rng.randint(25, 100))
            y1 = min(height - 1, y0 + rng.randint(25, 100))
            color = tuple(rng.randint(50, 220) for _ in range(3))
            cv2.rectangle(image, (x0, y0), (x1, y1), color, -1, cv2.LINE_AA)
    return finish_camera_effects(image, rng)


def main() -> None:
    args = parse_args()
    if args.samples_per_class <= 0:
        raise ValueError("--samples-per-class must be positive")
    if args.image_width < 128 or args.image_height < 128:
        raise ValueError("image size should be at least 128x128")
    if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1.0:
        raise ValueError("train/val ratios must leave a positive test split")

    if args.all_signs:
        sign_keys = sorted(DEFAULT_SIGNS)
    else:
        sign_keys = args.sign or ["turn_right"]
    selected_signs = [DEFAULT_SIGNS[key] for key in sign_keys]

    rng = random.Random(args.seed)
    np.random.seed(args.seed & 0xFFFFFFFF)

    ensure_clean_output_dir(args.output_dir, args.overwrite)
    for split in ("train", "val", "test"):
        (args.output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    write_dataset_metadata(args.output_dir, selected_signs, args)
    textures = [load_sign_texture(spec.texture_path) for spec in selected_signs]

    preview_paths: list[Path] = []
    total_positive = args.samples_per_class * len(selected_signs)
    print(f"Generating {total_positive} positive image(s) in {args.output_dir}")

    for class_id, (spec, texture_rgba) in enumerate(zip(selected_signs, textures)):
        for index in range(args.samples_per_class):
            split = split_name(index, args.samples_per_class, args.train_ratio, args.val_ratio)
            stem = f"{spec.key}_{index:05d}"
            image_path = args.output_dir / "images" / split / f"{stem}.{args.format}"
            label_path = args.output_dir / "labels" / split / f"{stem}.txt"

            image, bbox = render_one(
                texture_rgba,
                args.image_width,
                args.image_height,
                rng,
                min_object_px=args.min_object_px,
            )
            cv2.imwrite(str(image_path), image)
            label_path.write_text(
                yolo_line(class_id, bbox, args.image_width, args.image_height),
                encoding="utf-8",
            )
            if len(preview_paths) < 16:
                preview_paths.append(image_path)

    for index in range(args.negative_samples):
        split = split_name(index, args.negative_samples, args.train_ratio, args.val_ratio)
        stem = f"negative_{index:05d}"
        image_path = args.output_dir / "images" / split / f"{stem}.{args.format}"
        label_path = args.output_dir / "labels" / split / f"{stem}.txt"
        image = generate_negative_image(args.image_width, args.image_height, rng)
        cv2.imwrite(str(image_path), image)
        label_path.write_text("", encoding="utf-8")

    write_preview(args.output_dir, preview_paths)
    print(f"Wrote YOLO dataset: {args.output_dir}")
    print(f"Classes: {', '.join(spec.yolo_name for spec in selected_signs)}")
    print(f"Train with: yolo detect train data={args.output_dir / 'data.yaml'} model=yolov8n.pt imgsz={args.image_width}")


if __name__ == "__main__":
    main()
