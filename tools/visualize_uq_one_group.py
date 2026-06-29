"""Visualize one inference-time UQ group for paper figures.

The figure contains:
    multi-view camera input | raw LiDAR BEV | camera BEV feature
    LiDAR BEV feature      | camera UQ     | LiDAR UQ

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_uq_one_group.py \
        configs/nuscenes/seg/fusion_uq_phase2_flow_clear_day.yaml \
        runs/seg_uq_phase2_flow_clear_day/latest.pth \
        --condition night \
        --out-dir viz/uq_one_group
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from PIL import Image, ImageDraw, ImageFont
from torchpack.utils.config import configs

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
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", default="viz/uq_one_group")
    parser.add_argument("--ann-file", default=None)
    parser.add_argument(
        "--condition",
        choices=["all", "clear", "night", "rain"],
        default="night",
        help="Condition used for selecting a sample when --token is not set.",
    )
    parser.add_argument("--token", default=None)
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="Use the N-th sample matching --condition, useful when the first sample is visually weak.",
    )
    parser.add_argument(
        "--random",
        action="store_true",
        help="Randomly select one sample from the matched condition instead of using --rank.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--camera-width", type=int, default=260)
    parser.add_argument("--camera-height", type=int, default=150)
    parser.add_argument("--lidar-size", type=int, default=420)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument("--feature-cmap", default="viridis")
    parser.add_argument("--uq-cmap", default="magma")
    parser.add_argument("--feature-clip", nargs=2, type=float, default=[1.0, 99.0])
    parser.add_argument("--workers", type=int, default=0)
    args, opts = parser.parse_known_args()
    args.cfg_options = opts
    return args


def build_cfg(config: str, opts: list[str], ann_file: str | None) -> Config:
    configs.load(config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=config)
    cfg.model.train_cfg = None

    if hasattr(cfg.model, "uq_training"):
        cfg.model.uq_training = dict(enable=False, loss_weight=0.0)
    if hasattr(cfg.model, "flow_training"):
        cfg.model.flow_training = dict(enable=False, loss_weight=0.0)
    if ann_file is not None:
        cfg.data.test.ann_file = ann_file
    return cfg


def build_description_lookup(nuscenes_root: str, version: str) -> dict[str, str]:
    try:
        from nuscenes.nuscenes import NuScenes
    except Exception as exc:
        print(f"WARNING: cannot import nuScenes API: {exc}")
        return {}

    try:
        nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    except Exception as exc:
        print(f"WARNING: cannot read nuScenes metadata: {exc}")
        return {}

    token_to_desc = {}
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        token_to_desc[sample["token"]] = scene.get("description", "")
    return token_to_desc


def classify_scene(description: str) -> str:
    desc = description.lower()
    if any(keyword in desc for keyword in RAIN_KEYWORDS):
        return "rain"
    if any(keyword in desc for keyword in NIGHT_KEYWORDS):
        return "night"
    return "clear"


def select_one_info(
    data_infos: list[dict],
    token_to_desc: dict[str, str],
    condition: str,
    token: str | None,
    rank: int,
    random_select: bool,
    seed: int,
) -> tuple[dict, str, str]:
    if token is not None:
        for info in data_infos:
            if info["token"] == token:
                desc = token_to_desc.get(token, "")
                return info, classify_scene(desc) if desc else "all", desc
        raise RuntimeError(f"Token not found in ann_file: {token}")

    matches = []
    for info in data_infos:
        desc = token_to_desc.get(info["token"], "")
        label = classify_scene(desc) if desc else "all"
        if condition == "all" or label == condition:
            matches.append((info, label, desc))
    if not matches:
        raise RuntimeError(f"No samples matched condition={condition}")
    if random_select:
        rng = np.random.RandomState(seed)
        return matches[int(rng.randint(0, len(matches)))]
    if rank >= len(matches):
        raise RuntimeError(f"rank={rank} out of range; only {len(matches)} matches.")
    return matches[rank]


def register_feature_hooks(model):
    core = model.module if hasattr(model, "module") else model
    features = {}
    hooks = []

    def make_hook(name):
        def _hook(_module, _inputs, output):
            if torch.is_tensor(output):
                features[name] = output.detach().float().cpu()
        return _hook

    hooks.append(
        core.encoders["camera"]["vtransform"].register_forward_hook(
            make_hook("camera_bev")
        )
    )
    hooks.append(
        core.encoders["lidar"]["backbone"].register_forward_hook(
            make_hook("lidar_bev")
        )
    )
    return features, hooks


def grab_uq(model) -> tuple[np.ndarray, np.ndarray]:
    core = model.module if hasattr(model, "module") else model
    fuser = core.fuser
    if not hasattr(fuser, "last_uq_cam") or fuser.last_uq_cam is None:
        raise RuntimeError(
            f"Could not find UQ maps on fuser {type(fuser).__name__}. "
            "Use a UQ/flow config and checkpoint."
        )
    uq_cam = fuser.last_uq_cam.detach().float().cpu()[0, 0].numpy()
    uq_lid = fuser.last_uq_lid.detach().float().cpu()[0, 0].numpy()
    return uq_cam, uq_lid


def normalize_percentile(array: np.ndarray, clip: list[float]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(array, clip)
    array = np.clip(array, lo, hi)
    if hi > lo:
        return (array - lo) / (hi - lo)
    return np.zeros_like(array)


def feature_to_map(feature: torch.Tensor, clip: list[float]) -> np.ndarray:
    if feature.dim() == 4:
        feature = feature[0]
    if feature.dim() != 3:
        raise ValueError(f"Expected CxHxW or BxCxHxW feature, got {tuple(feature.shape)}")
    fmap = feature.abs().mean(dim=0).numpy()
    return normalize_percentile(fmap, clip)


def resolve_path(path: str, root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute():
        candidate = root / p
        if candidate.exists():
            return candidate
    return p


def camera_name_from_path(path: str) -> str | None:
    p = Path(path)
    for part in reversed(p.parts):
        if part.startswith("CAM_"):
            return part
    return None


def make_camera_grid(
    filenames: list[str],
    root: Path,
    cell_w: int,
    cell_h: int,
) -> Image.Image:
    path_by_cam = {}
    for filename in filenames:
        cam = camera_name_from_path(filename)
        if cam is not None:
            path_by_cam[cam] = resolve_path(filename, root)

    grid = Image.new("RGB", (cell_w * 3, cell_h * 2), (40, 40, 40))
    draw = ImageDraw.Draw(grid)
    try:
        font = ImageFont.truetype("Arial.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    for idx, (cam, label) in enumerate(CAMERA_ORDER):
        row, col = divmod(idx, 3)
        x0, y0 = col * cell_w, row * cell_h
        path = path_by_cam.get(cam)
        if path is None or not path.exists():
            image = Image.new("RGB", (cell_w, cell_h), (15, 15, 15))
        else:
            image = Image.open(path).convert("RGB").resize((cell_w, cell_h), Image.BILINEAR)
        # Dark title strip keeps labels readable without hiding too much context.
        strip = Image.new("RGBA", (cell_w, 30), (0, 0, 0, 110))
        image.paste(strip, (0, 0), strip)
        grid.paste(image, (x0, y0))
        draw.text((x0 + 8, y0 + 5), label, fill=(255, 255, 255), font=font)
    return grid


def render_lidar_bev(
    points: np.ndarray,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    size: int,
    max_points: int,
) -> Image.Image:
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    keep = (x >= xlim[0]) & (x <= xlim[1]) & (y >= ylim[0]) & (y <= ylim[1])
    points = points[keep]
    if points.shape[0] > max_points:
        rng = np.random.RandomState(0)
        points = points[rng.choice(points.shape[0], max_points, replace=False)]

    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    px = ((x - xlim[0]) / (xlim[1] - xlim[0]) * (size - 1)).astype(np.int32)
    py = ((ylim[1] - y) / (ylim[1] - ylim[0]) * (size - 1)).astype(np.int32)

    canvas = np.full((size, size, 3), 245, dtype=np.uint8)
    z_norm = np.clip((z + 3.0) / 6.0, 0.0, 1.0)
    colors = np.stack(
        [
            (30 + 190 * z_norm).astype(np.uint8),
            (70 + 130 * (1.0 - z_norm)).astype(np.uint8),
            np.full_like(z_norm, 220, dtype=np.uint8),
        ],
        axis=1,
    )
    canvas[py, px] = colors
    return Image.fromarray(canvas)


def save_array_image(
    array: np.ndarray,
    out_path: Path,
    cmap: str,
    title: str,
    vmin: float = 0.0,
    vmax: float = 1.0,
) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def add_panel_title(image: Image.Image, title: str) -> Image.Image:
    title_h = 34
    out = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    out.paste(image.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(20, 20, 20), font=font)
    return out


def resize_panel(image: Image.Image, width: int, height: int) -> Image.Image:
    return image.convert("RGB").resize((width, height), Image.BILINEAR)


def compose_group(
    camera_grid: Image.Image,
    lidar_input: Image.Image,
    cam_feat_path: Path,
    lid_feat_path: Path,
    uq_cam_path: Path,
    uq_lid_path: Path,
    out_path: Path,
    header: str,
) -> None:
    panel_w, panel_h = 360, 280
    panels = [
        add_panel_title(resize_panel(camera_grid, panel_w, panel_h), "Multi-view Camera"),
        add_panel_title(resize_panel(lidar_input, panel_w, panel_h), "LiDAR Input"),
        add_panel_title(resize_panel(Image.open(cam_feat_path), panel_w, panel_h), "Camera BEV Feature"),
        add_panel_title(resize_panel(Image.open(lid_feat_path), panel_w, panel_h), "LiDAR BEV Feature"),
        add_panel_title(resize_panel(Image.open(uq_cam_path), panel_w, panel_h), "Camera UQ"),
        add_panel_title(resize_panel(Image.open(uq_lid_path), panel_w, panel_h), "LiDAR UQ"),
    ]
    gap = 10
    header_h = 42
    cols, rows = 3, 2
    width = cols * panel_w + (cols - 1) * gap
    height = header_h + rows * (panel_h + 34) + (rows - 1) * gap
    canvas = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("Arial.ttf", 22)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 10), header, fill=(20, 20, 20), font=font)
    for idx, panel in enumerate(panels):
        row, col = divmod(idx, cols)
        x = col * (panel_w + gap)
        y = header_h + row * (panel_h + 34 + gap)
        canvas.paste(panel, (x, y))
    canvas.save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nusc_root = Path(args.nuscenes_root)

    cfg = build_cfg(args.config, args.cfg_options, args.ann_file)
    dataset_cfg = dict(cfg.data.test)
    dataset_cfg["test_mode"] = True
    dataset = build_dataset(dataset_cfg)

    token_to_desc = build_description_lookup(args.nuscenes_root, args.nuscenes_version)
    selected_info, condition, desc = select_one_info(
        dataset.data_infos,
        token_to_desc,
        args.condition,
        args.token,
        args.rank,
        args.random,
        args.seed,
    )
    dataset.data_infos = [selected_info]

    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=args.workers,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    features, hooks = register_feature_hooks(model)
    data = next(iter(loader))

    try:
        with torch.inference_mode():
            _ = model(return_loss=False, rescale=True, **data)
            uq_cam, uq_lid = grab_uq(model)
            if "camera_bev" not in features or "lidar_bev" not in features:
                raise RuntimeError("Feature hooks did not capture camera/lidar BEV features.")

            cam_feat = feature_to_map(features["camera_bev"], args.feature_clip)
            lid_feat = feature_to_map(features["lidar_bev"], args.feature_clip)
    finally:
        for hook in hooks:
            hook.remove()

    metas = data["metas"].data[0][0]
    camera_grid = make_camera_grid(
        metas["filename"],
        nusc_root,
        args.camera_width,
        args.camera_height,
    )
    points = data["points"].data[0][0].numpy()
    xlim = (float(cfg.point_cloud_range[0]), float(cfg.point_cloud_range[3]))
    ylim = (float(cfg.point_cloud_range[1]), float(cfg.point_cloud_range[4]))
    lidar_input = render_lidar_bev(points, xlim, ylim, args.lidar_size, args.max_points)

    token = selected_info["token"]
    stem = f"{condition}_{token}"
    camera_grid_path = out_dir / f"{stem}_camera_grid.png"
    lidar_input_path = out_dir / f"{stem}_lidar_input.png"
    cam_feat_path = out_dir / f"{stem}_camera_bev_feature.png"
    lid_feat_path = out_dir / f"{stem}_lidar_bev_feature.png"
    uq_cam_path = out_dir / f"{stem}_uq_cam.png"
    uq_lid_path = out_dir / f"{stem}_uq_lid.png"
    group_path = out_dir / f"{stem}_uq_group.png"

    camera_grid.save(camera_grid_path)
    lidar_input.save(lidar_input_path)
    save_array_image(cam_feat, cam_feat_path, args.feature_cmap, "Camera BEV feature")
    save_array_image(lid_feat, lid_feat_path, args.feature_cmap, "LiDAR BEV feature")
    save_array_image(uq_cam, uq_cam_path, args.uq_cmap, f"U_c mean={uq_cam.mean():.3f}")
    save_array_image(uq_lid, uq_lid_path, args.uq_cmap, f"U_l mean={uq_lid.mean():.3f}")
    compose_group(
        camera_grid,
        lidar_input,
        cam_feat_path,
        lid_feat_path,
        uq_cam_path,
        uq_lid_path,
        group_path,
        header=(
            f"Inference-time UQ visualization | condition={condition} | token={token}"
        ),
    )

    metadata = {
        "token": token,
        "condition": condition,
        "description": desc,
        "uq_cam_mean": float(uq_cam.mean()),
        "uq_lid_mean": float(uq_lid.mean()),
        "camera_grid": str(camera_grid_path),
        "lidar_input": str(lidar_input_path),
        "camera_bev_feature": str(cam_feat_path),
        "lidar_bev_feature": str(lid_feat_path),
        "uq_cam": str(uq_cam_path),
        "uq_lid": str(uq_lid_path),
        "group": str(group_path),
    }
    (out_dir / f"{stem}_metadata.json").write_text(json.dumps(metadata, indent=2))
    print(f"Saved group figure: {group_path}")
    print(f"Camera UQ mean={uq_cam.mean():.3f}, LiDAR UQ mean={uq_lid.mean():.3f}")


if __name__ == "__main__":
    main()
