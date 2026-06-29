"""Create nuScenes condition prediction panels.

This script selects samples from clear/day, night, and rain scenes, then renders
a compact panel for each sample:

    six camera views with predicted 3D boxes | LiDAR BEV with predicted boxes |
    predicted BEV map

Use a detection checkpoint for boxes and a segmentation checkpoint for maps.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import cv2
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from PIL import Image, ImageDraw, ImageFont
from torchpack.utils.config import configs

from mmdet3d.core import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


CAMERA_ORDER = [
    ("CAM_FRONT_LEFT", "Front Left"),
    ("CAM_FRONT", "Front"),
    ("CAM_FRONT_RIGHT", "Front Right"),
    ("CAM_BACK_LEFT", "Back Left"),
    ("CAM_BACK", "Back"),
    ("CAM_BACK_RIGHT", "Back Right"),
]

OBJECT_PALETTE = {
    "car": (255, 158, 0),
    "truck": (255, 99, 71),
    "construction_vehicle": (233, 150, 70),
    "bus": (255, 69, 0),
    "trailer": (255, 140, 0),
    "barrier": (112, 128, 144),
    "motorcycle": (255, 61, 99),
    "bicycle": (220, 20, 60),
    "pedestrian": (0, 0, 230),
    "traffic_cone": (47, 79, 79),
}

MAP_PALETTE = {
    "drivable_area": (166, 206, 227),
    "ped_crossing": (251, 154, 153),
    "walkway": (227, 26, 28),
    "stop_line": (253, 191, 111),
    "carpark_area": (255, 127, 0),
    "divider": (106, 61, 154),
}

NIGHT_KEYWORDS = ("night",)
RAIN_KEYWORDS = ("rain", "raining")


def recursive_eval(obj, globals=None):
    if globals is None:
        globals = copy.deepcopy(obj)

    if isinstance(obj, dict):
        for key in obj:
            obj[key] = recursive_eval(obj[key], globals)
    elif isinstance(obj, list):
        for key, val in enumerate(obj):
            obj[key] = recursive_eval(val, globals)
    elif isinstance(obj, str) and obj.startswith("${") and obj.endswith("}"):
        obj = eval(obj[2:-1], globals)
        obj = recursive_eval(obj, globals)

    return obj


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "config",
        help=(
            "Base config used for sample selection. If --det-config/--seg-config "
            "are omitted, this config is also used for inference."
        ),
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Optional single checkpoint. Used for both detection and segmentation if the model has both heads.",
    )
    parser.add_argument("--det-config", default=None, help="Detection model config.")
    parser.add_argument("--det-checkpoint", default=None, help="Detection checkpoint.")
    parser.add_argument("--seg-config", default=None, help="Segmentation model config.")
    parser.add_argument("--seg-checkpoint", default=None, help="Segmentation checkpoint.")
    parser.add_argument("--out-dir", default="viz/nuscenes_condition_panels")
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument(
        "--occluded-demo",
        action="store_true",
        help="Visualize one Dirt, one Water Blur, and one LiDAR spatial occlusion sample.",
    )
    parser.add_argument(
        "--dirt-ann-file",
        default="data/nuscenes/nuscenes_infos_val_camocc_dirt_0p3.pkl",
    )
    parser.add_argument(
        "--waterblur-ann-file",
        default="data/nuscenes/nuscenes_infos_val_camocc_waterblur_0p3.pkl",
    )
    parser.add_argument(
        "--spatial-ann-file",
        default="data/nuscenes/nuscenes_infos_val.pkl",
    )
    parser.add_argument("--spatial-region", default="front")
    parser.add_argument("--spatial-drop-percentage", type=int, default=100)
    parser.add_argument("--num-per-condition", type=int, default=2)
    parser.add_argument(
        "--selection",
        choices=["first", "rich"],
        default="rich",
        help=(
            "Sample selection strategy. 'rich' ranks samples by object count "
            "and category diversity using annotations only for token selection."
        ),
    )
    parser.add_argument(
        "--min-objects",
        type=int,
        default=8,
        help="Preferred minimum number of annotated objects when --selection rich.",
    )
    parser.add_argument("--bbox-score", type=float, default=0.3)
    parser.add_argument("--map-score", type=float, default=0.5)
    parser.add_argument("--camera-width", type=int, default=352)
    parser.add_argument("--camera-height", type=int, default=192)
    parser.add_argument("--bev-size", type=int, default=384)
    parser.add_argument("--condition-width", type=int, default=120)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--box-line-width", type=int, default=2)
    parser.add_argument(
        "--ann-file",
        default=None,
        help="Optional override for cfg.data.test.ann_file.",
    )
    args, opts = parser.parse_known_args()
    args.cfg_options = opts
    return args


def build_cfg(config: str, opts: list[str]) -> Config:
    configs.load(config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=config)
    return cfg


def build_inference_cfg(config: str, opts: list[str], ann_file: str | None) -> Config:
    cfg = build_cfg(config, opts)
    cfg.data.test["test_mode"] = True
    if ann_file is not None:
        cfg.data.test["ann_file"] = ann_file
    return cfg


def build_description_lookup(nuscenes_root: str, version: str):
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    token_to_desc = {}
    token_to_scene = {}
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        token_to_desc[sample["token"]] = scene.get("description", "")
        token_to_scene[sample["token"]] = sample["scene_token"]
    return token_to_desc, token_to_scene


def classify_scene(description: str) -> str:
    desc = description.lower()
    if any(keyword in desc for keyword in RAIN_KEYWORDS):
        return "rain"
    if any(keyword in desc for keyword in NIGHT_KEYWORDS):
        return "night"
    return "clear"


def select_indices(dataset, token_to_desc, token_to_scene, num_per_condition: int):
    selected: dict[str, list[int]] = {"clear": [], "night": [], "rain": []}
    scene_counts: dict[str, defaultdict[str, int]] = {
        key: defaultdict(int) for key in selected
    }

    for idx, info in enumerate(dataset.data_infos):
        token = info["token"]
        condition = classify_scene(token_to_desc.get(token, ""))
        if len(selected[condition]) >= num_per_condition:
            continue

        scene = token_to_scene.get(token, "")
        if scene_counts[condition][scene] >= 1:
            continue

        selected[condition].append(idx)
        scene_counts[condition][scene] += 1

        if all(len(values) >= num_per_condition for values in selected.values()):
            break

    for condition, values in selected.items():
        if len(values) < num_per_condition:
            print(
                f"WARNING: found only {len(values)} {condition} samples; "
                f"requested {num_per_condition}."
            )
    return selected


def object_stats(info: dict) -> tuple[int, int]:
    names = info.get("gt_names", None)
    boxes = info.get("gt_boxes", None)
    if names is not None:
        names = np.asarray(names)
        if "num_lidar_pts" in info:
            valid = np.asarray(info["num_lidar_pts"]) > 0
            if valid.shape[0] == names.shape[0]:
                names = names[valid]
        count = int(names.shape[0])
        diversity = int(len(set(names.tolist())))
        return count, diversity
    if boxes is not None:
        return int(np.asarray(boxes).shape[0]), 0
    return 0, 0


def select_rich_indices(
    dataset,
    token_to_desc,
    token_to_scene,
    num_per_condition: int,
    min_objects: int,
):
    candidates: dict[str, list[tuple[float, int, int, int]]] = {
        "clear": [],
        "night": [],
        "rain": [],
    }

    for idx, info in enumerate(dataset.data_infos):
        token = info["token"]
        condition = classify_scene(token_to_desc.get(token, ""))
        count, diversity = object_stats(info)
        if count == 0:
            continue
        rich_bonus = 20 if count >= min_objects else 0
        score = count + 3 * diversity + rich_bonus
        candidates[condition].append((float(score), count, diversity, idx))

    selected: dict[str, list[int]] = {"clear": [], "night": [], "rain": []}
    used_scenes: dict[str, set[str]] = {key: set() for key in selected}
    for condition, items in candidates.items():
        items = sorted(items, reverse=True)
        for _score, _count, _diversity, idx in items:
            scene = token_to_scene.get(dataset.data_infos[idx]["token"], "")
            if scene in used_scenes[condition]:
                continue
            selected[condition].append(idx)
            used_scenes[condition].add(scene)
            if len(selected[condition]) >= num_per_condition:
                break
        if len(selected[condition]) < num_per_condition:
            for _score, _count, _diversity, idx in items:
                if idx in selected[condition]:
                    continue
                selected[condition].append(idx)
                if len(selected[condition]) >= num_per_condition:
                    break

    for condition, values in selected.items():
        if len(values) < num_per_condition:
            print(
                f"WARNING: found only {len(values)} {condition} samples; "
                f"requested {num_per_condition}."
            )
    return selected


def select_rich_any_index(
    dataset,
    min_objects: int,
    used_tokens: set[str] | None = None,
) -> int:
    used_tokens = used_tokens or set()
    candidates = []
    for idx, info in enumerate(dataset.data_infos):
        if info["token"] in used_tokens:
            continue
        count, diversity = object_stats(info)
        if count == 0:
            continue
        rich_bonus = 20 if count >= min_objects else 0
        score = count + 3 * diversity + rich_bonus
        candidates.append((float(score), count, diversity, idx))
    if not candidates:
        raise RuntimeError("No annotated samples found for rich selection.")
    return sorted(candidates, reverse=True)[0][-1]


def build_single_dataset(config: str, opts: list[str], ann_file: str | None):
    cfg = build_inference_cfg(config, opts, ann_file)
    return build_dataset(cfg.data.test), cfg


def occluded_lidar_opts(args: argparse.Namespace) -> list[str]:
    return [
        "lidar_occlusion.mode=spatial",
        f"lidar_occlusion.drop_percentage={args.spatial_drop_percentage}",
        f"lidar_occlusion.region={args.spatial_region}",
        "lidar_occlusion.angle_range=90",
        "lidar_occlusion.seed=42",
    ]


def build_occluded_specs(args: argparse.Namespace):
    raw_specs = [
        {
            "condition": "Dirt",
            "ann_file": args.dirt_ann_file,
            "opts": [],
        },
        {
            "condition": "Water Blur",
            "ann_file": args.waterblur_ann_file,
            "opts": [],
        },
        {
            "condition": "Spatial Occl.",
            "ann_file": args.spatial_ann_file,
            "opts": occluded_lidar_opts(args),
        },
    ]

    specs = []
    used_tokens = set()
    for spec in raw_specs:
        dataset, _cfg = build_single_dataset(
            args.config,
            args.cfg_options + spec["opts"],
            spec["ann_file"],
        )
        idx = select_rich_any_index(dataset, args.min_objects, used_tokens)
        info = dataset.data_infos[idx]
        used_tokens.add(info["token"])
        count, diversity = object_stats(info)
        specs.append(
            {
                **spec,
                "token": info["token"],
                "info": info,
                "selection_object_count": count,
                "selection_category_count": diversity,
            }
        )
    return specs


def ordered_infos_by_token(dataset, tokens: list[str]) -> list[dict]:
    info_by_token = {info["token"]: info for info in dataset.data_infos}
    missing = [token for token in tokens if token not in info_by_token]
    if missing:
        raise KeyError(f"Tokens missing from dataset: {missing[:5]}")
    return [info_by_token[token] for token in tokens]


def collect_model_predictions(
    config: str | None,
    checkpoint: str | None,
    tokens: list[str],
    ann_file: str | None,
    opts: list[str],
):
    if config is None or checkpoint is None:
        return {}, None

    cfg = build_inference_cfg(config, opts, ann_file)
    dataset = build_dataset(cfg.data.test)
    dataset.data_infos = ordered_infos_by_token(dataset, tokens)
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    torch.cuda.set_device(0)
    model = build_model(cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    results = {}
    with torch.inference_mode():
        for data in dataloader:
            metas = data["metas"].data[0][0]
            token = metas["token"]
            outputs = model(return_loss=False, rescale=True, **data)
            result = dict(outputs[0])
            result["metas"] = metas
            if "points" in data:
                result["points"] = data["points"].data[0][0].numpy()
            results[token] = result

    del model
    torch.cuda.empty_cache()
    return results, cfg


def collect_scenario_predictions(
    config: str | None,
    checkpoint: str | None,
    specs: list[dict],
    base_opts: list[str],
):
    if config is None or checkpoint is None:
        return {}, None

    model_cfg = build_inference_cfg(config, base_opts, None)
    torch.cuda.set_device(0)
    model = build_model(model_cfg.model)
    load_checkpoint(model, checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    results = {}
    render_cfg = None
    with torch.inference_mode():
        for spec in specs:
            cfg = build_inference_cfg(
                config,
                base_opts + spec.get("opts", []),
                spec.get("ann_file", None),
            )
            render_cfg = render_cfg or cfg
            dataset = build_dataset(cfg.data.test)
            dataset.data_infos = ordered_infos_by_token(dataset, [spec["token"]])
            dataloader = build_dataloader(
                dataset,
                samples_per_gpu=1,
                workers_per_gpu=0,
                dist=False,
                shuffle=False,
            )
            for data in dataloader:
                metas = data["metas"].data[0][0]
                outputs = model(return_loss=False, rescale=True, **data)
                result = dict(outputs[0])
                result["metas"] = metas
                if "points" in data:
                    result["points"] = data["points"].data[0][0].numpy()
                results[spec["token"]] = result

    del model
    torch.cuda.empty_cache()
    return results, render_cfg or model_cfg


def prediction_boxes(result: dict, score_thr: float):
    if "boxes_3d" not in result:
        return None, None
    bboxes = result["boxes_3d"].tensor.numpy()
    scores = result["scores_3d"].numpy()
    labels = result["labels_3d"].numpy()
    keep = scores >= score_thr
    bboxes = bboxes[keep]
    labels = labels[keep]
    if bboxes.shape[0] == 0:
        return None, labels
    bboxes[..., 2] -= bboxes[..., 5] / 2
    return LiDARInstance3DBoxes(bboxes, box_dim=bboxes.shape[-1]), labels


def prediction_masks(result: dict, score_thr: float):
    if "masks_bev" not in result:
        return None
    masks = result["masks_bev"].numpy()
    return masks >= score_thr


def get_font(size: int):
    for name in ("Arial.ttf", "DejaVuSans.ttf", "/System/Library/Fonts/Supplemental/Arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def resolve_path(path: str, nuscenes_root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute():
        candidate = nuscenes_root / p
        if candidate.exists():
            return candidate
    return p


def draw_label_bar(image: Image.Image, text: str, font_size: int = 34) -> None:
    draw = ImageDraw.Draw(image)
    font = get_font(font_size)
    bar_h = int(font_size * 1.35)
    draw.rectangle((0, 0, image.width, bar_h), fill=(96, 96, 96))
    draw.text((10, 2), text, fill=(255, 255, 255), font=font)


def boxes_to_numpy(bboxes):
    if bboxes is None or len(bboxes) == 0:
        return None
    corners = bboxes.corners
    if hasattr(corners, "detach"):
        corners = corners.detach().cpu().numpy()
    return corners


def render_camera(
    image_path: Path,
    bboxes,
    labels: np.ndarray,
    transform: np.ndarray,
    classes: list[str],
    title: str,
    size: tuple[int, int],
    line_width: int,
) -> Image.Image:
    image = mmcv.imread(str(image_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    canvas = image.copy()

    corners = boxes_to_numpy(bboxes)
    if corners is not None and corners.shape[0] > 0:
        num_bboxes = corners.shape[0]
        coords = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_bboxes * 8, 1))], axis=-1
        )
        transform = copy.deepcopy(transform).reshape(4, 4)
        coords = coords @ transform.T
        coords = coords.reshape(-1, 8, 4)

        visible = np.all(coords[..., 2] > 0, axis=1)
        coords = coords[visible]
        visible_labels = labels[visible]
        if coords.shape[0] > 0:
            order = np.argsort(-np.min(coords[..., 2], axis=1))
            coords = coords[order]
            visible_labels = visible_labels[order]

            coords = coords.reshape(-1, 4)
            coords[:, 2] = np.clip(coords[:, 2], 1e-5, 1e5)
            coords[:, 0] /= coords[:, 2]
            coords[:, 1] /= coords[:, 2]
            coords = coords[..., :2].reshape(-1, 8, 2)

            for box_index in range(coords.shape[0]):
                label = int(visible_labels[box_index])
                if label < 0 or label >= len(classes):
                    continue
                color = OBJECT_PALETTE.get(classes[label], (255, 158, 0))
                color_bgr = (int(color[2]), int(color[1]), int(color[0]))
                for start, end in [
                    (0, 1),
                    (0, 3),
                    (0, 4),
                    (1, 2),
                    (1, 5),
                    (3, 2),
                    (3, 7),
                    (4, 5),
                    (4, 7),
                    (2, 6),
                    (5, 6),
                    (6, 7),
                ]:
                    cv2.line(
                        canvas,
                        coords[box_index, start].astype(np.int32),
                        coords[box_index, end].astype(np.int32),
                        color_bgr,
                        line_width,
                        cv2.LINE_AA,
                    )

    image = Image.fromarray(canvas).resize(size, Image.BILINEAR)
    draw_label_bar(image, title, font_size=30)
    return image


def render_camera_grid(
    metas: dict,
    bboxes,
    labels: np.ndarray,
    classes: list[str],
    nuscenes_root: Path,
    camera_size: tuple[int, int],
    line_width: int,
) -> Image.Image:
    filenames = metas["filename"]
    transforms = metas["lidar2image"]

    camera_to_idx = {}
    for idx, path in enumerate(filenames):
        text = str(path)
        for camera_name, _title in CAMERA_ORDER:
            if f"/{camera_name}/" in text or f"__{camera_name}__" in text:
                camera_to_idx[camera_name] = idx
                break

    views = []
    for camera_name, title in CAMERA_ORDER:
        idx = camera_to_idx.get(camera_name)
        if idx is None:
            raise KeyError(f"Could not find {camera_name} in filenames: {filenames}")
        image_path = resolve_path(str(filenames[idx]), nuscenes_root)
        views.append(
            render_camera(
                image_path,
                bboxes,
                labels,
                transforms[idx],
                classes,
                title,
                camera_size,
                line_width,
            )
        )

    width, height = camera_size
    grid = Image.new("RGB", (width * 3, height * 2), (255, 255, 255))
    for idx, view in enumerate(views):
        x = (idx % 3) * width
        y = (idx // 3) * height
        grid.paste(view, (x, y))
    return grid


def lidar_to_pixel(
    xy: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    size: int,
) -> np.ndarray:
    px = (xy[:, 0] - xlim[0]) / (xlim[1] - xlim[0]) * (size - 1)
    py = (ylim[1] - xy[:, 1]) / (ylim[1] - ylim[0]) * (size - 1)
    return np.stack([px, py], axis=1)


def render_lidar(
    points: np.ndarray,
    bboxes,
    labels: np.ndarray,
    classes: list[str],
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    size: int,
    max_points: int,
    line_width: int,
) -> Image.Image:
    keep = (
        (points[:, 0] >= xlim[0])
        & (points[:, 0] <= xlim[1])
        & (points[:, 1] >= ylim[0])
        & (points[:, 1] <= ylim[1])
    )
    points = points[keep]
    if points.shape[0] > max_points:
        rng = np.random.RandomState(0)
        points = points[rng.choice(points.shape[0], max_points, replace=False)]

    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    if points.shape[0] > 0:
        xy = lidar_to_pixel(points[:, :2], xlim, ylim, size).astype(np.int32)
        z_norm = np.clip((points[:, 2] + 4.0) / 8.0, 0.0, 1.0)
        gray = (30 + 210 * z_norm).astype(np.uint8)
        canvas[xy[:, 1].clip(0, size - 1), xy[:, 0].clip(0, size - 1)] = np.stack(
            [gray, gray, gray], axis=1
        )

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)

    corners = boxes_to_numpy(bboxes)
    if corners is not None and corners.shape[0] > 0:
        poly_indices = [0, 3, 7, 4, 0]
        for box_index in range(corners.shape[0]):
            label = int(labels[box_index])
            if label < 0 or label >= len(classes):
                continue
            color = OBJECT_PALETTE.get(classes[label], (255, 158, 0))
            poly = lidar_to_pixel(corners[box_index, poly_indices, :2], xlim, ylim, size)
            points_xy = [(float(x), float(y)) for x, y in poly]
            draw.line(points_xy, fill=color, width=line_width)

    cx, cy = lidar_to_pixel(np.array([[0.0, 0.0]]), xlim, ylim, size)[0]
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(0, 0, 0))
    draw_label_bar(image, "LiDAR", font_size=36)
    return image


def render_map(masks: np.ndarray, classes: list[str], size: int) -> Image.Image:
    masks = masks.astype(bool)
    canvas = np.full((*masks.shape[-2:], 3), 240, dtype=np.uint8)
    for idx, name in enumerate(classes):
        if idx < masks.shape[0] and name in MAP_PALETTE:
            canvas[masks[idx]] = MAP_PALETTE[name]
    canvas = np.rot90(canvas, 1)
    image = Image.fromarray(canvas).resize((size, size), Image.NEAREST)
    draw_label_bar(image, "Map", font_size=36)
    return image


def make_sample_row(
    condition: str,
    metas: dict,
    points: np.ndarray,
    bboxes,
    labels: np.ndarray,
    masks: np.ndarray,
    cfg: Config,
    args: argparse.Namespace,
    nuscenes_root: Path,
) -> Image.Image:
    camera_grid = render_camera_grid(
        metas,
        bboxes,
        labels,
        cfg.object_classes,
        nuscenes_root,
        (args.camera_width, args.camera_height),
        args.box_line_width,
    )
    xlim = (float(cfg.point_cloud_range[0]), float(cfg.point_cloud_range[3]))
    ylim = (float(cfg.point_cloud_range[1]), float(cfg.point_cloud_range[4]))
    lidar = render_lidar(
        points,
        bboxes,
        labels,
        cfg.object_classes,
        xlim,
        ylim,
        args.bev_size,
        args.max_points,
        args.box_line_width,
    )
    map_image = render_map(masks, cfg.map_classes, args.bev_size)

    row_h = max(camera_grid.height, lidar.height, map_image.height)
    condition_w = args.condition_width
    row_w = condition_w + camera_grid.width + lidar.width + map_image.width
    row = Image.new("RGB", (row_w, row_h), (255, 255, 255))

    cond_panel = Image.new("RGB", (condition_w, row_h), (230, 230, 230))
    draw = ImageDraw.Draw(cond_panel)
    font = get_font(20)
    condition_text = condition.upper()
    try:
        text_w = draw.textbbox((0, 0), condition_text, font=font)[2]
    except AttributeError:
        text_w = draw.textsize(condition_text, font=font)[0]
    if text_w > condition_w - 18 and " " in condition_text:
        parts = condition_text.split(" ", 1)
        draw.text((10, 12), parts[0], fill=(20, 20, 20), font=font)
        draw.text((10, 38), parts[1], fill=(20, 20, 20), font=font)
    else:
        draw.text((10, 14), condition_text, fill=(20, 20, 20), font=font)
    row.paste(cond_panel, (0, 0))
    row.paste(camera_grid, (condition_w, 0))
    row.paste(lidar, (condition_w + camera_grid.width, 0))
    row.paste(map_image, (condition_w + camera_grid.width + lidar.width, 0))
    return row


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nuscenes_root = Path(args.nuscenes_root)

    det_config = args.det_config or args.config
    det_checkpoint = args.det_checkpoint or args.checkpoint
    seg_config = args.seg_config or args.config
    seg_checkpoint = args.seg_checkpoint or args.checkpoint

    if det_checkpoint is None and seg_checkpoint is None:
        raise ValueError(
            "Prediction visualization requires --checkpoint, or "
            "--det-checkpoint/--seg-checkpoint."
        )

    token_to_desc, token_to_scene = build_description_lookup(
        args.nuscenes_root, args.nuscenes_version
    )
    if args.occluded_demo:
        specs = build_occluded_specs(args)
        selected_tokens = [spec["token"] for spec in specs]
        ordered_conditions = [spec["condition"] for spec in specs]
        selected_infos = [spec["info"] for spec in specs]
        det_results, det_cfg = collect_scenario_predictions(
            det_config,
            det_checkpoint,
            specs,
            args.cfg_options,
        )
        seg_results, seg_cfg = collect_scenario_predictions(
            seg_config,
            seg_checkpoint,
            specs,
            args.cfg_options,
        )
        render_cfg = det_cfg or seg_cfg or build_inference_cfg(
            args.config, args.cfg_options, args.ann_file
        )
        summary_name = "summary_occluded_failures.png"
    else:
        cfg = build_inference_cfg(args.config, args.cfg_options, args.ann_file)
        dataset_cfg = copy.deepcopy(cfg.data.test)
        dataset_cfg["test_mode"] = True
        dataset = build_dataset(dataset_cfg)
        if args.selection == "rich":
            selected = select_rich_indices(
                dataset,
                token_to_desc,
                token_to_scene,
                args.num_per_condition,
                args.min_objects,
            )
        else:
            selected = select_indices(
                dataset,
                token_to_desc,
                token_to_scene,
                args.num_per_condition,
            )

        ordered_indices: list[int] = []
        ordered_conditions: list[str] = []
        for condition in ("clear", "night", "rain"):
            for idx in selected[condition][: args.num_per_condition]:
                ordered_indices.append(idx)
                ordered_conditions.append(condition)

        if not ordered_indices:
            raise RuntimeError("No samples selected. Check nuScenes root/version and ann_file.")

        selected_infos = [dataset.data_infos[idx] for idx in ordered_indices]
        selected_tokens = [info["token"] for info in selected_infos]

        det_results, det_cfg = collect_model_predictions(
            det_config,
            det_checkpoint,
            selected_tokens,
            args.ann_file,
            args.cfg_options,
        )
        seg_results, seg_cfg = collect_model_predictions(
            seg_config,
            seg_checkpoint,
            selected_tokens,
            args.ann_file,
            args.cfg_options,
        )
        render_cfg = det_cfg or seg_cfg or cfg
        summary_name = "summary_clear_night_rain_2each.png"

    rows = []
    metadata = []
    for row_idx, token in enumerate(selected_tokens):
        condition = ordered_conditions[row_idx]
        info = selected_infos[row_idx]
        desc = token_to_desc.get(token, "")
        det_result = det_results.get(token, {})
        seg_result = seg_results.get(token, {})
        source_result = det_result or seg_result
        if not source_result:
            raise RuntimeError(f"No prediction result found for token {token}")

        metas = source_result["metas"]
        points = source_result.get("points", None)
        if points is None:
            raise RuntimeError(
                "The selected model pipeline did not return LiDAR points; "
                "cannot render the LiDAR panel."
            )
        bboxes, labels = prediction_boxes(det_result, args.bbox_score)
        masks = prediction_masks(seg_result, args.map_score)
        if masks is None:
            raise RuntimeError(
                "No predicted BEV map found. Pass --seg-config and --seg-checkpoint "
                "for a segmentation model."
            )

        row = make_sample_row(
            condition,
            metas,
            points,
            bboxes,
            labels,
            masks,
            render_cfg,
            args,
            nuscenes_root,
        )
        rows.append(row)

        sample_name = f"{row_idx + 1:02d}_{condition}_{metas['timestamp']}-{token}"
        sample_path = out_dir / f"{sample_name}.png"
        row.save(sample_path)
        metadata.append(
            {
                "condition": condition,
                "token": token,
                "scene_token": token_to_scene.get(token, ""),
                "description": desc,
                "selection_object_count": object_stats(info)[0],
                "selection_category_count": object_stats(info)[1],
                "ann_file": specs[row_idx]["ann_file"] if args.occluded_demo else args.ann_file,
                "extra_options": specs[row_idx].get("opts", []) if args.occluded_demo else [],
                "det_checkpoint": det_checkpoint,
                "seg_checkpoint": seg_checkpoint,
                "output": str(sample_path),
            }
        )
        count, diversity = object_stats(info)
        print(
            f"[{condition}] {token} | objects={count}, classes={diversity} | {desc}"
        )

    total_w = max(row.width for row in rows)
    total_h = sum(row.height for row in rows)
    summary = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    y = 0
    for row in rows:
        summary.paste(row, (0, y))
        y += row.height
    summary_path = out_dir / summary_name
    summary.save(summary_path)

    (out_dir / "selected_samples.json").write_text(json.dumps(metadata, indent=2))
    (out_dir / "selected_tokens.txt").write_text(
        "\n".join(item["token"] for item in metadata) + "\n"
    )
    print(f"Saved summary: {summary_path}")
    print(f"Saved metadata: {out_dir / 'selected_samples.json'}")


if __name__ == "__main__":
    main()
