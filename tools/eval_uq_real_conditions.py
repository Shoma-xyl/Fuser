"""Evaluate the Phase-1 trained reliability head on REAL nuScenes
val samples stratified by scene description (clean-day / night / rain).

The question this script answers:

    Does the head, trained only on SYNTHETIC corruption (C1 darkening,
    C2 blur, etc.), actually respond to REAL sensor degradation (real
    night driving, real rain)? If not, Phase 2 + UOT completion is
    pointless because uq_cam will stay at the clean-sample baseline
    (~0.4) even in night/rain, so no transport will happen.

Method
------
1. Load nuScenes val set. For each sample, look up its scene description
   via the NuScenes SDK.
2. Classify samples into {clean_day, night, rain} by keyword matching
   in the description string. Skip ambiguous.
3. Run the (Phase-1) model forward on N samples from each bucket with
   NO synthetic corruption — just real data.
4. Collect mean(uq_cam), mean(uq_lid), and their spatial patterns.
5. Report aggregate stats + per-sample distributions.

PASS / NO-PASS criterion (for whether Phase 2 is worth running):

    night_mean_uq_cam - clean_day_mean_uq_cam  >  0.15     [PASS cam]
    rain_mean_uq_cam  - clean_day_mean_uq_cam  >  0.10     [PASS cam]
    (both lid uq should stay roughly at clean baseline)

If PASS: head generalizes to real conditions, Phase 2 UOT will do work.
If NO-PASS: head overfitted to synthetic corruption distribution. We'd
need to augment Phase 1 with a few real night/rain samples (domain-
randomization-style) before Phase 2.

Usage
-----
    python tools/eval_uq_real_conditions.py \
        configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_uq_phase1.yaml \
        runs/uq_phase1/latest.pth \
        --out-dir output/uq_real_conditions \
        --nuscenes-root data/nuscenes \
        --nuscenes-version v1.0-trainval \
        --num-per-bucket 15

Arguments
---------
--num-per-bucket : how many samples to evaluate from each of the 3
    buckets (clean_day, night, rain). 15 is a reasonable default: gives
    stable means without running for too long. Rain samples are rarer
    in nuScenes (~5% of val) so >30 may not be available.
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


# ---------------------------------------------------------------------------
# Scene classification by description text
# ---------------------------------------------------------------------------

# Keywords that indicate each condition. Tuned against nuScenes
# description strings. If a description matches BOTH night and rain, we
# classify as 'rain' (since rain is rarer — preserve its samples).
NIGHT_KEYWORDS = ["night"]
RAIN_KEYWORDS = ["rain", "raining"]


def classify_scene(description: str) -> str:
    """Return one of {"clean_day", "night", "rain", "ambiguous"}."""
    d = description.lower()
    has_rain = any(k in d for k in RAIN_KEYWORDS)
    has_night = any(k in d for k in NIGHT_KEYWORDS)
    if has_rain:
        return "rain"
    if has_night:
        return "night"
    return "clean_day"


# ---------------------------------------------------------------------------
# Model loading and inference utilities (adapted from sanity_uq_learned.py)
# ---------------------------------------------------------------------------

def build_model_and_loader(cfg_path: str, ckpt_path: str):
    configs.load(cfg_path, recursive=True)
    cfg = Config(recursive_eval(configs), filename=cfg_path)

    ds_cfg = dict(cfg.data.test)
    ds_cfg["test_mode"] = True
    dataset = build_dataset(ds_cfg)
    dl = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )

    cfg.model.train_cfg = None
    # Disable uq_training so we don't apply synthetic corruption inside
    # the model forward. We want raw real-world inputs.
    if hasattr(cfg.model, "uq_training"):
        cfg.model.uq_training = dict(enable=False, loss_weight=0.0)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, ckpt_path, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return cfg, dataset, dl, model


def grab_uq(model) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pull the last forward's uq_cam / uq_lid from the UQFuser."""
    core = model.module if hasattr(model, "module") else model
    fuser = core.fuser
    assert hasattr(fuser, "last_uq_cam"), \
        f"Expected UQFuser; got {type(fuser).__name__}"
    return fuser.last_uq_cam.detach().cpu().clone(), \
           fuser.last_uq_lid.detach().cpu().clone()


# ---------------------------------------------------------------------------
# NuScenes SDK wrapper for looking up scene description from sample token
# ---------------------------------------------------------------------------

class SceneDescriptionLookup:
    """Maps sample_token -> scene.description via the NuScenes SDK.

    Loads the JSON metadata once, builds a fast dict for lookup.
    """

    def __init__(self, nusc_root: str, version: str = "v1.0-trainval"):
        from nuscenes.nuscenes import NuScenes
        print(f"[lookup] Initializing NuScenes SDK ({version}) at {nusc_root}...")
        self.nusc = NuScenes(version=version, dataroot=nusc_root, verbose=False)
        # Build token -> description map
        self._map = {}
        for sample in self.nusc.sample:
            scene = self.nusc.get("scene", sample["scene_token"])
            self._map[sample["token"]] = scene["description"]
        print(f"[lookup] Indexed {len(self._map)} samples")

    def __call__(self, sample_token: str) -> str:
        return self._map.get(sample_token, "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--nuscenes-root", default="data/nuscenes")
    p.add_argument("--nuscenes-version", default="v1.0-trainval")
    p.add_argument("--num-per-bucket", type=int, default=15,
                   help="Samples per {clean_day, night, rain}")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build scene-description lookup
    lookup = SceneDescriptionLookup(args.nuscenes_root, args.nuscenes_version)

    # Build model + dataloader
    cfg, dataset, dl, model = build_model_and_loader(args.config, args.checkpoint)

    # -----------------------------------------------------------------
    # PRE-FILTER: scan data_infos (no data loading!) to pick sample
    # indices for each bucket. This takes < 1 second since we only
    # look at the token string per sample and do string matching.
    # -----------------------------------------------------------------
    need_per_bucket = args.num_per_bucket
    bucket_indices = {"clean_day": [], "night": [], "rank_rain": [], "rain": []}
    bucket_indices.pop("rank_rain")   # placeholder cleanup
    scene_seen = {"clean_day": set(), "night": set(), "rain": set()}

    print(f"\n[pre-filter] Scanning {len(dataset.data_infos)} val samples "
          f"by description (no data load)")
    # We'd like samples from DIFFERENT scenes when possible, to avoid
    # 15 near-identical consecutive samples from one 40-sample scene.
    # So we pick at most 3 samples per scene per bucket.
    MAX_PER_SCENE = 3
    # Map data_info idx → scene_token for diversity tracking.
    # The nuScenes SDK gives this:
    nusc = lookup.nusc
    sample_to_scene = {s["token"]: s["scene_token"] for s in nusc.sample}

    bucket_scene_count = {
        "clean_day": defaultdict(int),
        "night": defaultdict(int),
        "rain": defaultdict(int),
    }

    for idx, info in enumerate(dataset.data_infos):
        token = info["token"]
        desc = lookup(token)
        bucket = classify_scene(desc)
        scene_tok = sample_to_scene.get(token, "")

        # Diversity constraint: at most MAX_PER_SCENE samples from
        # the same scene into the same bucket.
        if bucket_scene_count[bucket][scene_tok] >= MAX_PER_SCENE:
            continue

        if len(bucket_indices[bucket]) < need_per_bucket:
            bucket_indices[bucket].append(idx)
            bucket_scene_count[bucket][scene_tok] += 1
            scene_seen[bucket].add(scene_tok)

        if all(len(bucket_indices[k]) >= need_per_bucket for k in bucket_indices):
            break

    print(f"[pre-filter] Pre-selected (targeting {need_per_bucket} per bucket):")
    for k in ["clean_day", "night", "rain"]:
        print(f"  {k:10s}: {len(bucket_indices[k])} samples "
              f"from {len(scene_seen[k])} distinct scenes")

    total_selected = sum(len(v) for v in bucket_indices.values())
    if total_selected == 0:
        print("\nERROR: no samples selected. Check nuscenes_root / version.")
        return

    if min(len(v) for v in bucket_indices.values()) == 0:
        print("\nWARNING: at least one bucket is empty. Continuing with "
              "what we have — deltas may be missing.")

    # -----------------------------------------------------------------
    # FORWARD pass: only on selected indices. Use a Subset DataLoader
    # so data loading is efficient.
    # -----------------------------------------------------------------
    from torch.utils.data import Subset

    selected_indices = []
    selected_buckets = []   # parallel list for lookup after forward
    for bucket in ["clean_day", "night", "rain"]:
        for idx in bucket_indices[bucket]:
            selected_indices.append(idx)
            selected_buckets.append(bucket)

    subset = Subset(dataset, selected_indices)
    sub_dl = build_dataloader(
        subset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )

    buckets = {"clean_day": [], "night": [], "rain": []}
    print(f"\n[forward] Running model on {len(selected_indices)} selected samples\n")

    with torch.no_grad():
        for i, data in enumerate(sub_dl):
            metas = data["metas"].data[0][0]
            token = metas.get("token", "")
            desc = lookup(token)
            bucket = selected_buckets[i]  # what we intended it to be

            _ = model(return_loss=False, rescale=True, **data)
            uq_cam, uq_lid = grab_uq(model)   # (1, 1, H, W) each

            uc = uq_cam[0, 0].numpy()
            ul = uq_lid[0, 0].numpy()

            buckets[bucket].append({
                "token": token,
                "desc": desc,
                "uq_cam": uc,
                "uq_lid": ul,
                "cam_mean": float(uc.mean()),
                "cam_max":  float(uc.max()),
                "cam_std":  float(uc.std()),
                "lid_mean": float(ul.mean()),
                "lid_max":  float(ul.max()),
                "lid_std":  float(ul.std()),
            })

            print(f"  [{i+1:3d}/{len(selected_indices)}] bucket={bucket:10s} "
                  f"uq_cam={uc.mean():.3f}  uq_lid={ul.mean():.3f}  "
                  f"desc={desc[:55]}")

    print(f"\n[forward] Done. Bucket sizes:")
    for k, v in buckets.items():
        print(f"  {k:10s}: {len(v)}")

    if min(len(v) for v in buckets.values()) == 0:
        print("\nWARNING: at least one bucket is empty. Cannot compute deltas.")
        return

    # ----- Aggregate statistics -----
    print("\n" + "=" * 70)
    print("AGGREGATE — mean over samples in each bucket")
    print("=" * 70)
    print(f"  {'bucket':10s}   {'N':>3s}   {'uq_cam_mean':>11s}   "
          f"{'uq_cam_max':>10s}   {'uq_lid_mean':>11s}   {'uq_lid_max':>10s}")
    print("  " + "-" * 68)
    agg = {}
    for k in ["clean_day", "night", "rain"]:
        v = buckets[k]
        if not v:
            continue
        cam_m = np.mean([s["cam_mean"] for s in v])
        cam_x = np.mean([s["cam_max"]  for s in v])
        lid_m = np.mean([s["lid_mean"] for s in v])
        lid_x = np.mean([s["lid_max"]  for s in v])
        agg[k] = {"cam_m": cam_m, "cam_x": cam_x,
                  "lid_m": lid_m, "lid_x": lid_x, "n": len(v)}
        print(f"  {k:10s}   {len(v):3d}   {cam_m:11.4f}   {cam_x:10.4f}   "
              f"{lid_m:11.4f}   {lid_x:10.4f}")

    # ----- Deltas against clean_day -----
    if "clean_day" in agg:
        base_cam = agg["clean_day"]["cam_m"]
        base_lid = agg["clean_day"]["lid_m"]
        print("\n  Deltas (vs. clean_day mean):")
        for k in ["night", "rain"]:
            if k not in agg:
                continue
            d_cam = agg[k]["cam_m"] - base_cam
            d_lid = agg[k]["lid_m"] - base_lid
            print(f"    {k:10s}   Δ_cam = {d_cam:+.4f}   Δ_lid = {d_lid:+.4f}")

    # ----- Decision guidance -----
    print("\n" + "=" * 70)
    print("DECISION GUIDANCE (does head generalize to real conditions?)")
    print("=" * 70)
    if "night" in agg and "clean_day" in agg:
        d = agg["night"]["cam_m"] - agg["clean_day"]["cam_m"]
        print(f"  night  Δ_uq_cam = {d:+.4f}  "
              f"(want > +0.15 for real-night generalization)")
    if "rain" in agg and "clean_day" in agg:
        d = agg["rain"]["cam_m"] - agg["clean_day"]["cam_m"]
        print(f"  rain   Δ_uq_cam = {d:+.4f}  "
              f"(want > +0.10 for real-rain generalization)")
    print("=" * 70)

    # ----- Save per-sample stats as JSON -----
    json_rows = []
    for k, v in buckets.items():
        for s in v:
            json_rows.append({
                "bucket": k,
                "token": s["token"],
                "desc": s["desc"],
                "cam_mean": s["cam_mean"], "cam_max": s["cam_max"],
                "cam_std": s["cam_std"],
                "lid_mean": s["lid_mean"], "lid_max": s["lid_max"],
                "lid_std": s["lid_std"],
            })
    (out_dir / "per_sample_stats.json").write_text(json.dumps(json_rows, indent=2))
    print(f"\n[save] per-sample stats -> {out_dir / 'per_sample_stats.json'}")

    # ----- Per-bucket distribution histograms + representative samples -----
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    for col, k in enumerate(["clean_day", "night", "rain"]):
        v = buckets[k]
        if not v:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        cam_means = [s["cam_mean"] for s in v]
        lid_means = [s["lid_mean"] for s in v]
        axes[0, col].hist(cam_means, bins=10, range=(0, 1),
                          color="tab:red", alpha=0.7)
        axes[0, col].set_title(f"{k} (N={len(v)})  uq_cam mean")
        axes[0, col].set_xlim(0, 1); axes[0, col].set_xlabel("uq_cam mean per sample")
        axes[0, col].axvline(np.mean(cam_means), color="k", lw=2,
                             label=f"mean={np.mean(cam_means):.3f}")
        axes[0, col].legend()

        axes[1, col].hist(lid_means, bins=10, range=(0, 1),
                          color="tab:blue", alpha=0.7)
        axes[1, col].set_title(f"{k}  uq_lid mean")
        axes[1, col].set_xlim(0, 1); axes[1, col].set_xlabel("uq_lid mean per sample")
        axes[1, col].axvline(np.mean(lid_means), color="k", lw=2,
                             label=f"mean={np.mean(lid_means):.3f}")
        axes[1, col].legend()

    fig.suptitle("Per-sample uq distribution by real-world condition",
                 fontsize=12)
    fig.tight_layout()
    hist_path = out_dir / "distribution_by_condition.png"
    fig.savefig(hist_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] histograms           -> {hist_path}")

    # ----- Representative BEV maps: 2 samples from each bucket -----
    for k in ["clean_day", "night", "rain"]:
        v = buckets[k][:2]
        if not v:
            continue
        fig, axes = plt.subplots(len(v), 2, figsize=(8, 4 * len(v)),
                                  squeeze=False)
        for r, s in enumerate(v):
            axes[r, 0].imshow(s["uq_cam"], cmap="hot", vmin=0, vmax=1)
            axes[r, 0].set_title(f"uq_cam mean={s['cam_mean']:.3f}")
            axes[r, 0].set_xticks([]); axes[r, 0].set_yticks([])
            axes[r, 1].imshow(s["uq_lid"], cmap="hot", vmin=0, vmax=1)
            axes[r, 1].set_title(f"uq_lid mean={s['lid_mean']:.3f}")
            axes[r, 1].set_xticks([]); axes[r, 1].set_yticks([])
        fig.suptitle(f"{k} — example BEV uq maps", fontsize=12)
        fig.tight_layout()
        path = out_dir / f"examples_{k}.png"
        fig.savefig(path, dpi=110, bbox_inches="tight")
        plt.close(fig)
        print(f"[save] examples {k:10s} -> {path}")

    print("\nDone.")


if __name__ == "__main__":
    main()