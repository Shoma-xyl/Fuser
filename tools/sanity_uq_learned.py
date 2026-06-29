"""Sanity check for the learned reliability head after Phase 1 training.

Runs the model on a handful of val samples, applying each corruption
type deterministically and comparing the resulting uq maps to the clean
baseline. Saves per-sample PNGs and prints aggregate statistics.

DECISION RULE:

PASS (proceed to Phase 2):
  * Clean:  mean(uq_cam) < 0.1  AND  mean(uq_lid) < 0.1
  * C1/C2/C3 (cam corrupt):  mean(uq_cam) > 0.5   within-sample
  * C4/C5   (lid corrupt):   mean(uq_lid) > 0.5   within-sample
  * uq_lid stays low (< 0.15) on cam-corrupted samples
  * uq_cam stays low (< 0.15) on lid-corrupted samples

FAIL (debug before Phase 2):
  * Clean uq is already high (> 0.3) — head is overly pessimistic
  * Corrupted uq doesn't rise — head didn't learn
  * Both uq rise under any corruption — head learned a trivial mask

Usage
-----
    python tools/sanity_uq_learned.py \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_uq_phase1.yaml \
      runs/uq_phase1/latest.pth \
      --out-dir output/uq_phase1_sanity \
      --num-samples 5
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

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
from mmdet3d.models.reliability import corruption_augment as ca
from mmdet3d.utils import recursive_eval


FRONT_CAMS = ["CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"]
# Indices into metas["filename"] following nuScenes order
# [FRONT, FRONT_RIGHT, FRONT_LEFT, BACK, BACK_LEFT, BACK_RIGHT]
# For [FL, F, FR] display we want [2, 0, 1]
FRONT_IDX = [2, 0, 1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("config")
    p.add_argument("checkpoint")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--num-samples", type=int, default=5)
    return p.parse_args()


def build_everything(args):
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)

    # Test pipeline on val dataset.
    ds_cfg = dict(cfg.data.test)
    ds_cfg["test_mode"] = True
    dataset = build_dataset(ds_cfg)
    dl = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=0,
        dist=False, shuffle=False,
    )

    # IMPORTANT: build model with UQFuser (phase1 config already does that),
    # then load our trained weights. Don't enable uq_training during eval —
    # we want to control corruption manually.
    cfg.model.train_cfg = None
    # Force uq_training off for sanity: we'll apply corruption manually
    # and just read uq from the stashed fuser state.
    if hasattr(cfg.model, "uq_training"):
        cfg.model.uq_training = dict(enable=False, loss_weight=0.0)

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    if cfg.get("fp16") is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model = MMDataParallel(model, device_ids=[0])
    model.eval()
    return cfg, dataset, dl, model


def grab_uq_from_model(model) -> tuple[torch.Tensor, torch.Tensor]:
    """Pull last_uq_cam/last_uq_lid from the UQFuser."""
    core = model.module if hasattr(model, "module") else model
    fuser = core.fuser
    assert hasattr(fuser, "last_uq_cam"), (
        "Expected UQFuser; got " + type(fuser).__name__
    )
    return fuser.last_uq_cam.detach().clone(), fuser.last_uq_lid.detach().clone()


@torch.no_grad()
def run_one_forward(model, data, corrupt_img_fn=None, corrupt_pts_fn=None):
    """Corrupt inputs in-place (temporarily), run forward, return (uq_cam, uq_lid)."""
    # data['img'] is a DataContainer; data['points'] is DataContainer of list
    img_dc = data["img"]
    pts_dc = data["points"]

    # Take underlying tensors
    orig_img = img_dc.data[0].clone()       # (1, 6, 3, H, W)
    orig_pts_list = list(pts_dc.data[0])    # list of 1 tensor (N, 5)

    # Apply corruption
    if corrupt_img_fn is not None:
        img_dc.data[0][:] = corrupt_img_fn(img_dc.data[0])
    if corrupt_pts_fn is not None:
        new_pts = corrupt_pts_fn(pts_dc.data[0][0])
        pts_dc.data[0][0] = new_pts

    try:
        _ = model(return_loss=False, rescale=True, **data)
        uq_cam, uq_lid = grab_uq_from_model(model)
    finally:
        # restore (so next corruption applies to a clean copy)
        img_dc.data[0][:] = orig_img
        pts_dc.data[0][0] = orig_pts_list[0]

    return uq_cam, uq_lid


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg, dataset, dl, model = build_everything(args)

    # Deterministic RNG for corruption
    rng = torch.Generator()
    rng.manual_seed(0xC0FFEE)

    # Corruption factories: each returns a function that applies corruption
    # to the appropriate input tensor.
    def mk_cam(sub):
        return lambda img: ca.corrupt_camera(img[0], sub, rng).unsqueeze(0)
    def mk_lid(sub):
        return lambda pts: ca.corrupt_lidar(pts, sub, rng)

    corruption_cases = [
        ("clean", None, None),
        ("cam_C1_night",  mk_cam("C1"), None),
        ("cam_C2_blur",   mk_cam("C2"), None),
        ("cam_C3_drop",   mk_cam("C3"), None),
        ("lid_C4_sparse", None, mk_lid("C4")),
        ("lid_C5_noise",  None, mk_lid("C5")),
    ]

    # Aggregate statistics across samples
    agg = {name: {"uq_cam": [], "uq_lid": []} for name, _, _ in corruption_cases}

    for i, data in enumerate(dl):
        if i >= args.num_samples:
            break
        metas = data["metas"].data[0][0]
        token = metas.get("token", f"idx{i:06d}")
        image_paths = list(metas.get("filename", []))

        # Run the 6 cases and collect uq maps per case.
        uq_by_case = {}
        for name, cim, cpt in corruption_cases:
            uq_c, uq_l = run_one_forward(model, data,
                                         corrupt_img_fn=cim,
                                         corrupt_pts_fn=cpt)
            uq_by_case[name] = (uq_c[0, 0].cpu().numpy(),
                                 uq_l[0, 0].cpu().numpy())
            agg[name]["uq_cam"].append(float(uq_c.mean()))
            agg[name]["uq_lid"].append(float(uq_l.mean()))

        # Print per-sample summary
        print(f"\n[sanity] sample {i} token={token}")
        print(f"  {'case':16s}   uq_cam mean    uq_lid mean")
        for name, _, _ in corruption_cases:
            uc, ul = uq_by_case[name]
            print(f"  {name:16s}   {uc.mean():.4f}        {ul.mean():.4f}")

        # Save a figure: rows are cases, cols are [front cam raw, uq_cam BEV, uq_lid BEV]
        fig, axes = plt.subplots(len(corruption_cases), 3, figsize=(14, 2.5 * len(corruption_cases)))
        for row, (name, _, _) in enumerate(corruption_cases):
            uc_np, ul_np = uq_by_case[name]

            ax = axes[row, 0]
            if len(image_paths) > 0 and os.path.exists(image_paths[0]):
                img = mmcv.imread(image_paths[0])[:, :, ::-1]
                ax.imshow(img)
            ax.set_title(f"{name} — CAM_FRONT", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

            ax = axes[row, 1]
            im = ax.imshow(uc_np, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"uq_cam BEV  mean={uc_np.mean():.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)

            ax = axes[row, 2]
            im = ax.imshow(ul_np, cmap="hot", vmin=0, vmax=1)
            ax.set_title(f"uq_lid BEV  mean={ul_np.mean():.3f}", fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            plt.colorbar(im, ax=ax, fraction=0.046)

        fig.suptitle(f"token={token}", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, 0.98))
        out_png = out_dir / f"{token}_uq_sanity.png"
        fig.savefig(out_png, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out_png}")

    # Aggregate summary
    print("\n" + "=" * 78)
    print("AGGREGATE SUMMARY (mean over samples)")
    print("=" * 78)
    print(f"  {'case':16s}   mean uq_cam    mean uq_lid")
    print("-" * 50)
    for name, _, _ in corruption_cases:
        ucs = np.array(agg[name]["uq_cam"])
        uls = np.array(agg[name]["uq_lid"])
        print(f"  {name:16s}   {ucs.mean():.4f}         {uls.mean():.4f}")

    # Decision rule
    print()
    clean_uc = np.mean(agg["clean"]["uq_cam"])
    clean_ul = np.mean(agg["clean"]["uq_lid"])
    cam_cases = ["cam_C1_night", "cam_C2_blur", "cam_C3_drop"]
    lid_cases = ["lid_C4_sparse", "lid_C5_noise"]
    cam_uc_mean = np.mean([np.mean(agg[c]["uq_cam"]) for c in cam_cases])
    cam_ul_mean = np.mean([np.mean(agg[c]["uq_lid"]) for c in cam_cases])
    lid_ul_mean = np.mean([np.mean(agg[c]["uq_lid"]) for c in lid_cases])
    lid_uc_mean = np.mean([np.mean(agg[c]["uq_cam"]) for c in lid_cases])

    print("DECISION GUIDANCE")
    print("-" * 78)
    print(f"  Clean uq_cam    = {clean_uc:.3f}   (want < 0.10)")
    print(f"  Clean uq_lid    = {clean_ul:.3f}   (want < 0.10)")
    print(f"  Cam-corrupt uq_cam = {cam_uc_mean:.3f}  (want > 0.50)")
    print(f"  Cam-corrupt uq_lid = {cam_ul_mean:.3f}  (want stay < 0.15)")
    print(f"  Lid-corrupt uq_lid = {lid_ul_mean:.3f}  (want > 0.50)")
    print(f"  Lid-corrupt uq_cam = {lid_uc_mean:.3f}  (want stay < 0.15)")

    pass_criteria = [
        clean_uc < 0.10,
        clean_ul < 0.10,
        cam_uc_mean > 0.50,
        cam_ul_mean < 0.15,
        lid_ul_mean > 0.50,
        lid_uc_mean < 0.15,
    ]
    if all(pass_criteria):
        print("\n  >>> PASS — proceed to Phase 2.")
    else:
        failed = [c for c, ok in zip(
            ["clean_cam_low", "clean_lid_low", "cam_cam_high",
             "cam_lid_stays_low", "lid_lid_high", "lid_cam_stays_low"],
            pass_criteria,
        ) if not ok]
        print(f"\n  >>> NOT PASS — failed criteria: {failed}")
        print("      Common causes:")
        print("        - Phase 1 not trained enough (retrain longer)")
        print("        - Loss weight too small")
        print("        - Corruption too subtle (head can't detect)")

    print("=" * 78)


if __name__ == "__main__":
    main()
