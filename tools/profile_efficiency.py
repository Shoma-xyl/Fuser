"""Profile Params, latency, and MACs for BEVFusion-style models.

The script uses real samples from the configured test dataloader:
  - Params are counted from model parameters.
  - Latency is measured as model inference time with batch size 1, excluding
    dataloader time.
  - MACs are profiled with the repo's THOP-based custom counters. By default,
    MACs are measured on one validation sample, matching the common
    "single-inference MACs" reporting protocol.

Example:
    CUDA_VISIBLE_DEVICES=0 python tools/profile_efficiency.py \
      configs/nuscenes/det/transfusion/secfpn/camera+lidar/swint_v0p075/convfuser_uq_phase2_flow.yaml \
      runs/uq_phase2_flow_w015_cp060_ep5/latest.pth \
      --latency-samples 200 \
      --mac-samples 1 \
      --cfg-options data.workers_per_gpu=8
"""
from __future__ import annotations

import argparse
import traceback
import warnings

import numpy as np
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint, wrap_fp16_model
from torch import nn
from torchpack.utils.config import configs

from mmdet3d.datasets import build_dataloader, build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import recursive_eval


class ProfileWrapper(nn.Module):
    """Wrap a data-parallel model so THOP can run a kwargs-based forward."""

    def __init__(self, model: MMDataParallel) -> None:
        super().__init__()
        self.model = model
        self.data = None

    def set_data(self, data: dict) -> None:
        self.data = data

    def forward(self, dummy: torch.Tensor):  # noqa: D401
        assert self.data is not None
        del dummy
        inputs, kwargs = self.model.scatter(
            (),
            dict(return_loss=False, rescale=True, **self.data),
            self.model.device_ids,
        )
        with torch.no_grad():
            return self.model.module(*inputs[0], **kwargs[0])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile inference efficiency.")
    parser.add_argument("config", help="Config file path.")
    parser.add_argument("checkpoint", help="Checkpoint file path.")
    parser.add_argument(
        "--latency-samples",
        type=int,
        default=200,
        help="Number of post-warmup samples used for latency averaging.",
    )
    parser.add_argument(
        "--mac-samples",
        type=int,
        default=1,
        help="Number of post-warmup samples used for average MACs.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
        help="Number of warmup iterations skipped before measuring latency.",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=50,
        help="Progress logging interval for measured latency samples.",
    )
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Force fp16 wrapper. The script also honors cfg.fp16 by default.",
    )
    parser.add_argument(
        "--skip-macs",
        action="store_true",
        help="Only measure parameters and latency.",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        action=DictAction,
        help="Override config options, e.g. data.workers_per_gpu=8.",
    )
    return parser.parse_args()


def count_params(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def load_cfg(args: argparse.Namespace) -> Config:
    configs.load(args.config, recursive=True)
    cfg = Config(recursive_eval(configs), filename=args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    return cfg


def main() -> None:
    args = parse_args()
    if args.latency_samples <= 0:
        raise ValueError("--latency-samples must be positive")
    if args.mac_samples < 0:
        raise ValueError("--mac-samples must be non-negative")

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(0)

    cfg = load_cfg(args)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False,
    )

    model = build_model(cfg.model, test_cfg=cfg.get("test_cfg"))
    fp16_cfg = cfg.get("fp16", None)
    if args.fp16 or fp16_cfg is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.cuda()
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    total_params, trainable_params = count_params(model.module)
    print("== Model Parameters ==")
    print(f"Params:           {total_params / 1e6:.3f} M")
    print(f"Trainable params: {trainable_params / 1e6:.3f} M")

    mac_values: list[float] = []
    profiler = ProfileWrapper(model) if not args.skip_macs and args.mac_samples > 0 else None
    flops_counter = None
    if profiler is not None:
        try:
            from mmdet3d.models.utils.flops_counter import flops_counter as _flops_counter

            flops_counter = _flops_counter
        except Exception as exc:  # pragma: no cover - depends on server env
            warnings.warn(
                "MAC profiling is disabled because flops_counter could not be imported. "
                f"Reason: {type(exc).__name__}: {exc}"
            )
            profiler = None
    dummy = torch.empty(1, device="cuda")

    measured = 0
    pure_inf_time = 0.0
    target_iters = args.warmup + args.latency_samples
    mac_data = None

    print("== Latency / MACs ==")
    for i, data in enumerate(data_loader):
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        with torch.no_grad():
            model(return_loss=False, rescale=True, **data)
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0

        if i >= args.warmup:
            if mac_data is None and profiler is not None and args.mac_samples > 0:
                mac_data = data

            if measured < args.latency_samples:
                pure_inf_time += elapsed
                measured += 1
                if measured % args.log_interval == 0:
                    latency_ms = pure_inf_time / measured * 1000.0
                    fps = 1000.0 / latency_ms
                    print(
                        f"Measured {measured}/{args.latency_samples} latency samples: "
                        f"{latency_ms:.2f} ms, {fps:.2f} FPS"
                    )

        if i + 1 >= target_iters:
            break

    if measured == 0:
        raise RuntimeError("No latency samples were measured. Reduce --warmup.")

    latency_ms = pure_inf_time / measured * 1000.0
    fps = 1000.0 / latency_ms

    if profiler is not None and flops_counter is not None and mac_data is not None:
        profiler.set_data(mac_data)
        try:
            for idx in range(args.mac_samples):
                macs, _ = flops_counter(profiler, (dummy,))
                mac_values.append(float(macs))
                print(
                    f"Profiled MAC sample {idx + 1}/{args.mac_samples}: "
                    f"{macs / 1e9:.3f} G"
                )
        except Exception as exc:  # pragma: no cover - server/runtime dependent
            warnings.warn(
                "MAC profiling failed. "
                f"Reason: {type(exc).__name__}: {exc}\n"
                + traceback.format_exc()
            )

    print("== Summary ==")
    print(f"Params (M):     {total_params / 1e6:.3f}")
    print(f"Latency (ms):   {latency_ms:.3f}")
    print(f"FPS:            {fps:.3f}")
    if args.skip_macs:
        print("MACs (G):       skipped")
    elif mac_values:
        mac_mean = float(np.mean(mac_values))
        mac_std = float(np.std(mac_values))
        print(f"MACs (G):       {mac_mean / 1e9:.3f}")
        if len(mac_values) > 1:
            print(f"MACs std (G):   {mac_std / 1e9:.3f}")
    else:
        print("MACs (G):       failed")


if __name__ == "__main__":
    main()
