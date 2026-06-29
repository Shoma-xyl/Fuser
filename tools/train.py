import argparse
import os
import random
import time

import numpy as np
import torch
import torch.distributed as torch_dist
from mmcv import Config
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs

from mmdet3d.apis import train_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import get_root_logger, convert_sync_batchnorm, recursive_eval

# Fix NCCL issues on cloud platforms (autodl etc.)
os.environ.setdefault("NCCL_P2P_DISABLE", "1")
os.environ.setdefault("NCCL_IB_DISABLE", "1")

# Single GPU:
#   CUDA_VISIBLE_DEVICES=0 python tools/train.py config.yaml --run-dir runs/xxx
# Multi-GPU (DDP):
#   CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --nproc_per_node=4 tools/train.py config.yaml --run-dir runs/xxx


def main():
    # Auto-detect distributed (torchrun sets these env vars)
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ

    if distributed:
        torch_dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        rank = int(os.environ["RANK"])
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        torch.cuda.set_device(0)

    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    parser.add_argument("--run-dir", metavar="DIR", help="run directory")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)

    cfg = Config(recursive_eval(configs), filename=args.config)

    torch.backends.cudnn.benchmark = cfg.cudnn_benchmark

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)
    cfg.run_dir = args.run_dir

    # dump config
    if rank == 0:
        cfg.dump(os.path.join(cfg.run_dir, "configs.yaml"))

    # init the logger before other steps
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    log_file = os.path.join(cfg.run_dir, f"{timestamp}.log")
    logger = get_root_logger(log_file=log_file)

    # log some basic info
    if rank == 0:
        try:
            logger.info(f"Config:\n{cfg.pretty_text}")
        except Exception:
            logger.info(f"Config:\n{cfg.text}")
        logger.info(
            f"Distributed: {distributed}, "
            f"GPUs: {int(os.environ.get('WORLD_SIZE', 1))}"
        )

    # set random seeds
    if cfg.seed is not None:
        logger.info(
            f"Set random seed to {cfg.seed}, "
            f"deterministic mode: {cfg.deterministic}"
        )
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if cfg.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    datasets = [build_dataset(cfg.data.train)]

    model = build_model(cfg.model)
    model.init_weights()
    if cfg.get("sync_bn", None):
        if not isinstance(cfg["sync_bn"], dict):
            cfg["sync_bn"] = dict(exclude=[])
        model = convert_sync_batchnorm(model, exclude=cfg["sync_bn"]["exclude"])

    if rank == 0:
        logger.info(f"Model:\n{model}")
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=True,
        timestamp=timestamp,
    )

    if distributed:
        torch_dist.destroy_process_group()


if __name__ == "__main__":
    main()
