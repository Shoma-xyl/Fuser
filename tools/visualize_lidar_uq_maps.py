"""Visualize LiDAR BEV features and LiDAR uncertainty maps for screening.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/visualize_lidar_uq_maps.py \
        configs/nuscenes/seg/fusion_uq_phase2_flow_clear_day.yaml \
        runs/seg_uq_phase2_flow_clear_day/latest.pth \
        --out-dir viz/lidar_uq_screen \
        --num-samples 50 \
        --condition all
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import defaultdict
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
    parser = argparse.ArgumentParser(
        description="Save LiDAR BEV feature and LiDAR UQ map visualizations."
    )
    parser.add_argument("config")
    parser.add_argument("checkpoint")
    parser.add_argument("--out-dir", default="viz/lidar_uq_screen")
    parser.add_argument("--ann-file", default=None)
    parser.add_argument("--num-samples", type=int, default=50)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument(
        "--condition",
        choices=["all", "clean", "night", "rain"],
        default="all",
        help="Subset samples by nuScenes scene description when metadata is available.",
    )
    parser.add_argument("--nuscenes-root", default="data/nuscenes")
    parser.add_argument("--nuscenes-version", default="v1.0-trainval")
    parser.add_argument("--tokens", nargs="+", default=None)
    parser.add_argument(
        "--token-file",
        default=None,
        help="Text file with one sample token per line. Used together with --tokens.",
    )
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--feature-cmap", default="viridis")
    parser.add_argument("--uq-cmap", default="magma")
    parser.add_argument("--feature-clip", nargs=2, type=float, default=[1.0, 99.0])
    parser.add_argument("--uq-clip", nargs=2, type=float, default=[1.0, 99.0])
    parser.add_argument(
        "--high-uq-percentile",
        type=float,
        default=85.0,
        help="Percentile threshold used for the high-UQ overlay in screening templates.",
    )
    parser.add_argument(
        "--feature-percentile",
        type=float,
        default=75.0,
        help="Percentile threshold used for the feature-support overlay in screening templates.",
    )
    parser.add_argument(
        "--no-template",
        action="store_true",
        help="Disable the per-sample screening template image.",
    )
    parser.add_argument("--save-npy", action="store_true")
    args, opts = parser.parse_known_args()
    args.cfg_options = opts
    return args


def classify_scene(description: str) -> str:
    desc = description.lower()
    if any(keyword in desc for keyword in RAIN_KEYWORDS):
        return "rain"
    if any(keyword in desc for keyword in NIGHT_KEYWORDS):
        return "night"
    return "clean"


def build_description_lookup(nuscenes_root: str, version: str) -> dict[str, str]:
    try:
        from nuscenes.nuscenes import NuScenes
    except Exception as exc:
        print(f"WARNING: cannot import nuScenes API, condition filtering disabled: {exc}")
        return {}

    try:
        nusc = NuScenes(version=version, dataroot=nuscenes_root, verbose=False)
    except Exception as exc:
        print(f"WARNING: cannot read nuScenes metadata, condition filtering disabled: {exc}")
        return {}

    token_to_desc = {}
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        token_to_desc[sample["token"]] = scene.get("description", "")
    return token_to_desc


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


def collect_tokens(args: argparse.Namespace) -> set[str] | None:
    tokens = []
    if args.tokens is not None:
        tokens.extend(args.tokens)
    if args.token_file is not None:
        token_path = Path(args.token_file)
        tokens.extend(
            line.strip()
            for line in token_path.read_text().splitlines()
            if line.strip() and not line.strip().startswith("#")
        )
    return set(tokens) if tokens else None


def select_data_infos(
    data_infos: list[dict],
    token_to_desc: dict[str, str],
    token_set: set[str] | None,
    condition: str,
    start: int,
    stride: int,
    num_samples: int,
) -> tuple[list[dict], list[str]]:
    selected = []
    labels = []
    seen_per_scene_desc = defaultdict(int)

    for index, info in enumerate(data_infos):
        token = info["token"]
        if token_set is not None and token not in token_set:
            continue
        desc = token_to_desc.get(token, "")
        label = classify_scene(desc) if desc else "all"
        if condition != "all" and label != condition:
            continue
        if index < start or (index - start) % max(stride, 1) != 0:
            continue

        # Avoid accidentally picking many adjacent frames from one condition when
        # scene metadata is available, while still keeping deterministic order.
        key = desc or "unknown"
        seen_per_scene_desc[key] += 1
        selected.append(info)
        labels.append(label)
        if len(selected) >= num_samples:
            break

    return selected, labels


def register_lidar_feature_hook(model):
    core = model.module if hasattr(model, "module") else model
    features = {}

    def hook_feature(_module, _inputs, output):
        if torch.is_tensor(output):
            features["lidar_bev"] = output.detach().float().cpu()

    hook = core.encoders["lidar"]["backbone"].register_forward_hook(hook_feature)
    return features, hook


def grab_lidar_uq(model) -> np.ndarray:
    core = model.module if hasattr(model, "module") else model
    fuser = core.fuser
    if not hasattr(fuser, "last_uq_lid") or fuser.last_uq_lid is None:
        raise RuntimeError(
            f"Could not find last_uq_lid on fuser {type(fuser).__name__}. "
            "Use a UQ/flow config and checkpoint."
        )
    return fuser.last_uq_lid.detach().float().cpu()[0, 0].numpy()


def normalize_percentile(array: np.ndarray, clip: list[float]) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    lo, hi = np.percentile(array, clip)
    array = np.clip(array, lo, hi)
    if hi > lo:
        array = (array - lo) / (hi - lo)
    else:
        array = np.zeros_like(array)
    return array


def feature_to_map(feature: torch.Tensor, clip: list[float]) -> np.ndarray:
    if feature.dim() == 4:
        feature = feature[0]
    if feature.dim() != 3:
        raise ValueError(f"Expected CxHxW or BxCxHxW feature, got {tuple(feature.shape)}")

    fmap = feature.abs().mean(dim=0).numpy()
    return normalize_percentile(fmap, clip)


def save_heatmap(
    array: np.ndarray,
    out_path: Path,
    cmap: str,
    title: str,
    clip: list[float] | None = None,
    add_colorbar: bool = False,
) -> np.ndarray:
    vis = normalize_percentile(array, clip) if clip is not None else array
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    im = ax.imshow(vis, cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    if add_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout(pad=0.1)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return vis


def add_label(image: Image.Image, text: str) -> Image.Image:
    label_h = 30
    out = Image.new("RGB", (image.width, image.height + label_h), (255, 255, 255))
    out.paste(image.convert("RGB"), (0, label_h))
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("Arial.ttf", 16)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 7), text, fill=(20, 20, 20), font=font)
    return out


def save_panel(feature_path: Path, uq_path: Path, out_path: Path, title: str) -> None:
    feat = Image.open(feature_path).convert("RGB")
    uq = Image.open(uq_path).convert("RGB")
    h = max(feat.height, uq.height)
    feat = feat.resize((int(feat.width * h / feat.height), h), Image.BILINEAR)
    uq = uq.resize((int(uq.width * h / uq.height), h), Image.BILINEAR)
    feat = add_label(feat, "LiDAR BEV Feature")
    uq = add_label(uq, "LiDAR UQ Map")
    title_h = 34
    out = Image.new(
        "RGB",
        (feat.width + uq.width, max(feat.height, uq.height) + title_h),
        (255, 255, 255),
    )
    draw = ImageDraw.Draw(out)
    try:
        font = ImageFont.truetype("Arial.ttf", 17)
    except OSError:
        font = ImageFont.load_default()
    draw.text((8, 8), title, fill=(20, 20, 20), font=font)
    out.paste(feat, (0, title_h))
    out.paste(uq, (feat.width, title_h))
    out.save(out_path)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    b = np.asarray(b, dtype=np.float32).reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.sqrt(np.square(a).sum() * np.square(b).sum()))
    if denom <= 1e-12:
        return 0.0
    return float((a * b).sum() / denom)


def edge_mean(array: np.ndarray, frac: float = 0.15) -> float:
    h, w = array.shape
    bh = max(1, int(round(h * frac)))
    bw = max(1, int(round(w * frac)))
    mask = np.zeros((h, w), dtype=bool)
    mask[:bh, :] = True
    mask[-bh:, :] = True
    mask[:, :bw] = True
    mask[:, -bw:] = True
    return float(array[mask].mean())


def center_mean(array: np.ndarray, frac: float = 0.50) -> float:
    h, w = array.shape
    ch = max(1, int(round(h * frac)))
    cw = max(1, int(round(w * frac)))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return float(array[y0 : y0 + ch, x0 : x0 + cw].mean())


def make_rgb_heatmap(array: np.ndarray, cmap_name: str) -> np.ndarray:
    cmap = plt.get_cmap(cmap_name)
    rgb = cmap(np.clip(array, 0.0, 1.0))[..., :3]
    return (rgb * 255.0).astype(np.uint8)


def save_screening_template(
    feature_map: np.ndarray,
    uq_raw: np.ndarray,
    uq_vis: np.ndarray,
    out_path: Path,
    title: str,
    feature_cmap: str,
    uq_cmap: str,
    feature_percentile: float,
    high_uq_percentile: float,
) -> dict[str, float]:
    feature_vis = np.asarray(feature_map, dtype=np.float32)
    uq = np.asarray(uq_raw, dtype=np.float32)

    high_uq_thr = float(np.percentile(uq, high_uq_percentile))
    feature_thr = float(np.percentile(feature_vis, feature_percentile))
    high_uq = uq >= high_uq_thr
    feature_support = feature_vis >= feature_thr
    overlap = high_uq & feature_support

    stats = {
        "uq_mean": float(uq.mean()),
        "uq_std": float(uq.std()),
        "uq_p75": float(np.percentile(uq, 75)),
        "uq_p90": float(np.percentile(uq, 90)),
        "uq_p95": float(np.percentile(uq, 95)),
        "uq_high_ratio": float(high_uq.mean()),
        "feature_support_ratio": float(feature_support.mean()),
        "overlap_ratio": float(overlap.sum() / max(float(high_uq.sum()), 1.0)),
        "feature_uq_corr": pearson_corr(feature_vis, uq),
        "edge_uq_mean": edge_mean(uq),
        "center_uq_mean": center_mean(uq),
    }
    stats["edge_center_gap"] = stats["edge_uq_mean"] - stats["center_uq_mean"]

    feature_rgb = make_rgb_heatmap(feature_vis, feature_cmap)
    uq_rgb = make_rgb_heatmap(uq_vis, uq_cmap)

    overlay = feature_rgb.copy().astype(np.float32)
    red = np.array([255, 35, 35], dtype=np.float32)
    overlay[high_uq] = 0.45 * overlay[high_uq] + 0.55 * red
    overlay = overlay.astype(np.uint8)

    support = np.zeros((*feature_vis.shape, 3), dtype=np.uint8)
    support[feature_support] = np.array([50, 150, 255], dtype=np.uint8)
    support[high_uq] = np.array([255, 70, 70], dtype=np.uint8)
    support[overlap] = np.array([255, 230, 70], dtype=np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 8.0))
    axes = axes.reshape(-1)
    panels = [
        (feature_rgb, "LiDAR BEV feature"),
        (uq_rgb, "LiDAR UQ map"),
        (overlay, f"High UQ overlay (top {100 - high_uq_percentile:.0f}%)"),
        (support, "Feature support / High UQ / Overlap"),
    ]
    for ax, (image, label) in zip(axes, panels):
        ax.imshow(image)
        ax.set_title(label, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])

    stat_text = (
        f"mean={stats['uq_mean']:.3f}  std={stats['uq_std']:.3f}  "
        f"p90={stats['uq_p90']:.3f}  p95={stats['uq_p95']:.3f}\n"
        f"corr(feature,UQ)={stats['feature_uq_corr']:.3f}  "
        f"overlap={stats['overlap_ratio']:.3f}  "
        f"edge-center={stats['edge_center_gap']:.3f}\n"
        "blue: feature support, red: high UQ, yellow: overlap"
    )
    fig.suptitle(title, fontsize=11, y=0.98)
    fig.text(0.5, 0.015, stat_text, ha="center", va="bottom", fontsize=9)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.955])
    fig.savefig(out_path, dpi=180, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return stats


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build_cfg(args.config, args.cfg_options, args.ann_file)
    dataset_cfg = dict(cfg.data.test)
    dataset_cfg["test_mode"] = True
    dataset = build_dataset(dataset_cfg)

    token_to_desc = build_description_lookup(args.nuscenes_root, args.nuscenes_version)
    token_set = collect_tokens(args)
    selected_infos, selected_labels = select_data_infos(
        dataset.data_infos,
        token_to_desc,
        token_set,
        args.condition,
        args.start,
        args.stride,
        args.num_samples,
    )
    if not selected_infos:
        raise RuntimeError("No samples selected. Check --condition, --tokens, or ann_file.")

    dataset.data_infos = selected_infos
    data_loader = build_dataloader(
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

    captured_features, hook = register_lidar_feature_hook(model)
    metadata = []

    try:
        with torch.inference_mode():
            for idx, data in enumerate(data_loader):
                captured_features.clear()
                info = selected_infos[idx]
                token = info["token"]
                condition = selected_labels[idx]
                stem = f"{idx:03d}_{condition}_{token}"

                _ = model(return_loss=False, rescale=True, **data)
                if "lidar_bev" not in captured_features:
                    raise RuntimeError("LiDAR BEV feature hook did not capture output.")

                uq_lid = grab_lidar_uq(model)
                lidar_bev = feature_to_map(captured_features["lidar_bev"], args.feature_clip)

                feature_path = out_dir / f"{stem}_lidar_bev_feature.png"
                uq_path = out_dir / f"{stem}_lidar_uq.png"
                panel_path = out_dir / f"{stem}_panel.png"
                template_path = out_dir / f"{stem}_template.png"

                save_heatmap(
                    lidar_bev,
                    feature_path,
                    cmap=args.feature_cmap,
                    title="LiDAR BEV feature",
                    clip=None,
                )
                uq_vis = save_heatmap(
                    uq_lid,
                    uq_path,
                    cmap=args.uq_cmap,
                    title=f"LiDAR UQ mean={uq_lid.mean():.3f}",
                    clip=args.uq_clip,
                )
                save_panel(
                    feature_path,
                    uq_path,
                    panel_path,
                    title=f"{condition} | token={token}",
                )
                template_stats = {}
                if not args.no_template:
                    template_stats = save_screening_template(
                        lidar_bev,
                        uq_lid,
                        uq_vis,
                        template_path,
                        title=f"{condition} | token={token}",
                        feature_cmap=args.feature_cmap,
                        uq_cmap=args.uq_cmap,
                        feature_percentile=args.feature_percentile,
                        high_uq_percentile=args.high_uq_percentile,
                    )

                if args.save_npy:
                    np.save(out_dir / f"{stem}_lidar_bev_feature.npy", lidar_bev)
                    np.save(out_dir / f"{stem}_lidar_uq.npy", uq_lid)

                row = {
                    "index": idx,
                    "token": token,
                    "condition": condition,
                    "description": token_to_desc.get(token, ""),
                    "uq_lid_mean": float(uq_lid.mean()),
                    "uq_lid_min": float(uq_lid.min()),
                    "uq_lid_max": float(uq_lid.max()),
                    "uq_lid_vis_mean": float(uq_vis.mean()),
                    "lidar_bev_feature": str(feature_path),
                    "lidar_uq": str(uq_path),
                    "panel": str(panel_path),
                }
                row.update(template_stats)
                if not args.no_template:
                    row["template"] = str(template_path)
                metadata.append(row)
                print(
                    f"[{idx + 1}/{len(selected_infos)}] {condition} token={token} "
                    f"uq_mean={uq_lid.mean():.3f} saved={template_path if not args.no_template else panel_path}"
                )
    finally:
        hook.remove()

    metadata_path = out_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(metadata)} samples to {out_dir}")
    print(f"Metadata: {metadata_path}")


if __name__ == "__main__":
    main()
