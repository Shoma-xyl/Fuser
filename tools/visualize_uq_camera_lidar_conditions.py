"""Visualize raw camera, raw LiDAR BEV, UQ maps, and BEV features.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_uq_camera_lidar_conditions.py \
        configs/nuscenes/seg/fusion_uq_phase2_flow_clear_day.yaml \
        runs/latest.pth \
        --out-dir viz/uq_camera_lidar_conditions \
        --num-per-condition 2 \
        --camera CAM_FRONT
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model


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
    parser.add_argument("--out-dir", default="viz/uq_camera_lidar_conditions")
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--num-per-condition", type=int, default=2)
    parser.add_argument("--camera", default="CAM_FRONT")
    parser.add_argument("--bev-size", type=int, default=900)
    parser.add_argument("--max-points", type=int, default=120000)
    parser.add_argument(
        "--save-panel",
        action="store_true",
        help="Also save a camera/LiDAR/UQ summary panel. Disabled by default.",
    )
    args, opts = parser.parse_known_args()
    args.cfg_options = opts
    return args


def classify_scene(description: str) -> str:
    desc = description.lower()
    if any(keyword in desc for keyword in RAIN_KEYWORDS):
        return "rain"
    if any(keyword in desc for keyword in NIGHT_KEYWORDS):
        return "night"
    return "clean_day"


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


def build_cfg(config: str, opts: list[str]) -> Config:
    configs.load(config, recursive=True)
    configs.update(opts)
    cfg = Config(recursive_eval(configs), filename=config)
    cfg.model.train_cfg = None
    if hasattr(cfg.model, "uq_training"):
        cfg.model.uq_training = dict(enable=False, loss_weight=0.0)
    if hasattr(cfg.model, "flow_training"):
        cfg.model.flow_training = dict(enable=False, loss_weight=0.0)
    return cfg


def grab_uq(model):
    core = model.module if hasattr(model, "module") else model
    fuser = core.fuser
    if not hasattr(fuser, "last_uq_cam") or fuser.last_uq_cam is None:
        raise RuntimeError(
            f"Could not find UQ maps on fuser {type(fuser).__name__}. "
            "Use a checkpoint/config with UQFuser or UQFlowFuser."
        )
    uq_cam = fuser.last_uq_cam.detach().float().cpu()[0, 0].numpy()
    uq_lid = fuser.last_uq_lid.detach().float().cpu()[0, 0].numpy()
    return uq_cam, uq_lid


def register_feature_hooks(model):
    core = model.module if hasattr(model, "module") else model
    features = {}
    hooks = []

    def hook_feature(name):
        def _hook(_module, _inputs, output):
            if torch.is_tensor(output):
                features[name] = output.detach().float().cpu()
        return _hook

    if "camera" in core.encoders:
        hooks.append(
            core.encoders["camera"]["vtransform"].register_forward_hook(
                hook_feature("camera_bev")
            )
        )
    if "lidar" in core.encoders:
        hooks.append(
            core.encoders["lidar"]["backbone"].register_forward_hook(
                hook_feature("lidar_bev")
            )
        )
    if core.fuser is not None:
        hooks.append(core.fuser.register_forward_hook(hook_feature("fused_bev")))

    return features, hooks


def resolve_path(path: str, nuscenes_root: Path) -> Path:
    p = Path(path)
    if p.exists():
        return p
    if not p.is_absolute():
        root_path = nuscenes_root / p
        if root_path.exists():
            return root_path
    return p


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
            (40 + 180 * z_norm).astype(np.uint8),
            (90 + 120 * (1.0 - z_norm)).astype(np.uint8),
            np.full_like(z_norm, 220, dtype=np.uint8),
        ],
        axis=1,
    )
    canvas[py, px] = colors

    image = Image.fromarray(canvas)
    draw = ImageDraw.Draw(image)
    cx = int((0.0 - xlim[0]) / (xlim[1] - xlim[0]) * (size - 1))
    cy = int((ylim[1] - 0.0) / (ylim[1] - ylim[0]) * (size - 1))
    draw.ellipse((cx - 5, cy - 5, cx + 5, cy + 5), fill=(220, 30, 30))
    draw.line((cx, cy, cx, max(0, cy - 45)), fill=(220, 30, 30), width=3)
    return image


def save_heatmap(array: np.ndarray, out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(array, cmap="hot", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def feature_to_map(feature: torch.Tensor) -> np.ndarray:
    if feature.dim() == 4:
        feature = feature[0]
    if feature.dim() != 3:
        raise ValueError(f"Expected CxHxW or BxCxHxW BEV feature, got {feature.shape}")

    fmap = feature.abs().mean(dim=0).numpy()
    lo, hi = np.percentile(fmap, [1.0, 99.0])
    fmap = np.clip(fmap, lo, hi)
    if hi > lo:
        fmap = (fmap - lo) / (hi - lo)
    else:
        fmap = np.zeros_like(fmap)
    return fmap


def save_feature_map(feature: torch.Tensor, out_path: Path, title: str) -> None:
    fmap = feature_to_map(feature)
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    im = ax.imshow(fmap, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def add_title(image: Image.Image, title: str) -> Image.Image:
    title_h = 38
    out = Image.new("RGB", (image.width, image.height + title_h), (255, 255, 255))
    out.paste(image.convert("RGB"), (0, title_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("Arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()
    draw.text((10, 10), title, fill=(20, 20, 20), font=font)
    return out


def resize_to_height(image: Image.Image, height: int) -> Image.Image:
    width = int(round(image.width * height / image.height))
    return image.resize((width, height), Image.BILINEAR)


def save_panel(paths: dict[str, Path], out_path: Path) -> None:
    panels = [
        add_title(resize_to_height(Image.open(paths["camera"]), 300), "Camera"),
        add_title(resize_to_height(Image.open(paths["lidar"]), 300), "LiDAR BEV"),
        add_title(resize_to_height(Image.open(paths["uq_cam"]), 300), "Camera UQ"),
        add_title(resize_to_height(Image.open(paths["uq_lid"]), 300), "LiDAR UQ"),
    ]
    width = sum(panel.width for panel in panels)
    height = max(panel.height for panel in panels)
    out = Image.new("RGB", (width, height), (255, 255, 255))
    x = 0
    for panel in panels:
        out.paste(panel, (x, 0))
        x += panel.width
    out.save(out_path)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nuscenes_root = Path(args.nuscenes_root)

    token_to_desc, token_to_scene = build_description_lookup(
        args.nuscenes_root, args.nuscenes_version
    )
    cfg = build_cfg(args.config, args.cfg_options)

    dataset_cfg = dict(cfg.data.test)
    dataset_cfg["test_mode"] = True
    dataset = build_dataset(dataset_cfg)

    selected = {"night": [], "rain": []}
    scene_count = {"night": defaultdict(int), "rain": defaultdict(int)}
    for idx, info in enumerate(dataset.data_infos):
        token = info["token"]
        bucket = classify_scene(token_to_desc.get(token, ""))
        if bucket not in selected:
            continue
        scene = token_to_scene.get(token, "")
        if scene_count[bucket][scene] >= 1:
            continue
        selected[bucket].append(idx)
        scene_count[bucket][scene] += 1
        if all(len(v) >= args.num_per_condition for v in selected.values()):
            break

    selected_indices = []
    selected_buckets = []
    for bucket in ["night", "rain"]:
        if len(selected[bucket]) < args.num_per_condition:
            print(
                f"WARNING: found only {len(selected[bucket])} {bucket} samples; "
                f"requested {args.num_per_condition}."
            )
        for idx in selected[bucket][: args.num_per_condition]:
            selected_indices.append(idx)
            selected_buckets.append(bucket)

    if not selected_indices:
        raise RuntimeError("No night/rain samples found. Check ann_file and nuScenes root.")

    selected_infos = [dataset.data_infos[idx] for idx in selected_indices]
    dataset.data_infos = selected_infos
    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    captured_features, hooks = register_feature_hooks(model)

    camera_names = list(selected_infos[0]["cams"].keys())
    if args.camera not in camera_names:
        raise KeyError(f"Camera {args.camera} not found. Available: {camera_names}")
    camera_idx = camera_names.index(args.camera)

    xlim = (float(cfg.point_cloud_range[0]), float(cfg.point_cloud_range[3]))
    ylim = (float(cfg.point_cloud_range[1]), float(cfg.point_cloud_range[4]))
    rows = []

    try:
        with torch.no_grad():
            for i, data in enumerate(dataloader):
                captured_features.clear()
                bucket = selected_buckets[i]
                info = selected_infos[i]
                token = info["token"]
                desc = token_to_desc.get(token, "")
                prefix = f"{bucket}_{len([r for r in rows if r['condition'] == bucket]) + 1}"

                _ = model(return_loss=False, rescale=True, **data)
                uq_cam, uq_lid = grab_uq(model)

                metas = data["metas"].data[0][0]
                camera_path = Path(metas["filename"][camera_idx])
                if not camera_path.exists():
                    camera_path = resolve_path(str(camera_path), nuscenes_root)
                lidar_points = data["points"].data[0][0].numpy()

                condition_dir = out_dir / bucket
                condition_dir.mkdir(parents=True, exist_ok=True)

                camera_out = condition_dir / f"{prefix}_{args.camera}_raw.jpg"
                lidar_out = condition_dir / f"{prefix}_lidar_raw_bev.png"
                uq_cam_out = condition_dir / f"{prefix}_uq_cam.png"
                uq_lid_out = condition_dir / f"{prefix}_uq_lid.png"
                cam_bev_out = condition_dir / f"{prefix}_camera_bev_feature.png"
                lid_bev_out = condition_dir / f"{prefix}_lidar_bev_feature.png"
                fused_bev_out = condition_dir / f"{prefix}_fused_bev_feature.png"
                panel_out = condition_dir / f"{prefix}_panel.png"

                Image.open(camera_path).convert("RGB").save(camera_out)
                render_lidar_bev(
                    lidar_points,
                    xlim=xlim,
                    ylim=ylim,
                    size=args.bev_size,
                    max_points=args.max_points,
                ).save(lidar_out)
                save_heatmap(uq_cam, uq_cam_out, f"{bucket} uq_cam mean={uq_cam.mean():.3f}")
                save_heatmap(uq_lid, uq_lid_out, f"{bucket} uq_lid mean={uq_lid.mean():.3f}")

                if "camera_bev" in captured_features:
                    save_feature_map(
                        captured_features["camera_bev"],
                        cam_bev_out,
                        f"{bucket} camera BEV feature",
                    )
                if "lidar_bev" in captured_features:
                    save_feature_map(
                        captured_features["lidar_bev"],
                        lid_bev_out,
                        f"{bucket} LiDAR BEV feature",
                    )
                if "fused_bev" in captured_features:
                    save_feature_map(
                        captured_features["fused_bev"],
                        fused_bev_out,
                        f"{bucket} fused BEV feature",
                    )

                if args.save_panel:
                    save_panel(
                        {
                            "camera": camera_out,
                            "lidar": lidar_out,
                            "uq_cam": uq_cam_out,
                            "uq_lid": uq_lid_out,
                        },
                        panel_out,
                    )

                row = {
                    "condition": bucket,
                    "token": token,
                    "description": desc,
                    "camera": args.camera,
                    "camera_path": str(camera_path),
                    "uq_cam_mean": float(uq_cam.mean()),
                    "uq_lid_mean": float(uq_lid.mean()),
                    "camera_raw": str(camera_out),
                    "lidar_raw_bev": str(lidar_out),
                    "uq_cam": str(uq_cam_out),
                    "uq_lid": str(uq_lid_out),
                }
                if cam_bev_out.exists():
                    row["camera_bev_feature"] = str(cam_bev_out)
                if lid_bev_out.exists():
                    row["lidar_bev_feature"] = str(lid_bev_out)
                if fused_bev_out.exists():
                    row["fused_bev_feature"] = str(fused_bev_out)
                if args.save_panel:
                    row["panel"] = str(panel_out)
                rows.append(row)

                print(
                    f"[{bucket}] token={token} "
                    f"uq_cam={uq_cam.mean():.3f} uq_lid={uq_lid.mean():.3f}"
                )
    finally:
        for hook in hooks:
            hook.remove()

    (out_dir / "metadata.json").write_text(json.dumps(rows, indent=2))
    print(f"Saved visualizations to {out_dir}")


if __name__ == "__main__":
    main()
