import torch
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import (
    DistSamplerSeedHook,
    EpochBasedRunner,
    GradientCumulativeFp16OptimizerHook,
    Fp16OptimizerHook,
    OptimizerHook,
    build_optimizer,
    build_runner,
)
from mmdet3d.runner import CustomEpochBasedRunner

from mmdet3d.utils import get_root_logger
from mmdet.core import DistEvalHook
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor


def train_model(
    model,
    dataset,
    cfg,
    distributed=False,
    validate=False,
    timestamp=None,
):
    logger = get_root_logger()

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]

    data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            None,
            dist=distributed,
            seed=cfg.seed,
        )
        for ds in dataset
    ]

    # put model on gpus
    find_unused_parameters = cfg.get("find_unused_parameters", False)
    # Sets the `find_unused_parameters` parameter in
    # torch.nn.parallel.DistributedDataParallel
    model = MMDistributedDataParallel(
        model.cuda(),
        device_ids=[torch.cuda.current_device()],
        broadcast_buffers=False,
        find_unused_parameters=find_unused_parameters,
    )

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    runner = build_runner(
        cfg.runner,
        default_args=dict(
            model=model,
            optimizer=optimizer,
            work_dir=cfg.run_dir,
            logger=logger,
            meta={},
        ),
    )
    
    if hasattr(runner, "set_dataset"):
        runner.set_dataset(dataset)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get("fp16", None)
    if fp16_cfg is not None:
        if "cumulative_iters" in cfg.optimizer_config:
            optimizer_config = GradientCumulativeFp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
        else:
            optimizer_config = Fp16OptimizerHook(
                **cfg.optimizer_config, **fp16_cfg, distributed=distributed
            )
    elif distributed and "type" not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get("momentum_config", None),
    )
    if isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        # Support batch_size > 1 in validation
        val_samples_per_gpu = cfg.data.val.pop("samples_per_gpu", 1)
        if val_samples_per_gpu > 1:
            # Replace 'ImageToTensor' to 'DefaultFormatBundle'
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False,
        )
        eval_cfg = cfg.get("evaluation", {})
        eval_cfg["by_epoch"] = cfg.runner["type"] != "IterBasedRunner"
        eval_hook = DistEvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # --- Remap fuser keys for checkpoint compatibility ---
        # Handles three loading scenarios:
        #   1. Pretrained BEVFusion → UQFuser(gate):
        #      fuser.0.* → fuser.base_fuser.0.*
        #   2. Pretrained BEVFusion → UQFuser(ot_attn):
        #      fuser.0.* → fuser.ot_attention.skip.0.*
        #   3. Phase 1 (gate) → Phase 2 (ot_attn):
        #      fuser.base_fuser.0.* → fuser.ot_attention.skip.0.*
        #   4. Pretrained/Phase 1 → UQFlowFuser:
        #      fuser.0.* or fuser.base_fuser.* →
        #      fuser.source_proj.* and fuser.teacher_proj.*
        import re
        _ckpt = torch.load(cfg.load_from, map_location="cpu")
        _sd = _ckpt["state_dict"] if "state_dict" in _ckpt else _ckpt

        _model = runner.model.module if hasattr(runner.model, "module") else runner.model
        _model_keys = set(_model.state_dict().keys())
        _has_base_fuser = any(k.startswith("fuser.base_fuser.") for k in _model_keys)
        _has_ot_attn = any(k.startswith("fuser.ot_attention.") for k in _model_keys)
        _has_flow_fuser = any(k.startswith("fuser.source_proj.") for k in _model_keys)

        _new_sd = {}
        _n_remapped = 0
        for k, v in _sd.items():
            if _has_flow_fuser:
                remapped = False
                if re.match(r'^fuser\.\d+', k):
                    _new_sd[re.sub(r'^fuser\.(\d+)', r'fuser.source_proj.\1', k)] = v
                    _new_sd[re.sub(r'^fuser\.(\d+)', r'fuser.teacher_proj.\1', k)] = v
                    _n_remapped += 2
                    remapped = True
                elif k.startswith("fuser.base_fuser."):
                    _new_sd[re.sub(r'^fuser\.base_fuser\.', r'fuser.source_proj.', k)] = v
                    _new_sd[re.sub(r'^fuser\.base_fuser\.', r'fuser.teacher_proj.', k)] = v
                    _n_remapped += 2
                    remapped = True
                if not remapped:
                    _new_sd[k] = v
                continue

            new_k = k
            if _has_ot_attn:
                # Scenario 3: Phase1 fuser.base_fuser.X → fuser.ot_attention.skip.X
                new_k = re.sub(r'^fuser\.base_fuser\.', r'fuser.ot_attention.skip.', new_k)
                # Scenario 2: pretrained fuser.X → fuser.ot_attention.skip.X
                new_k = re.sub(r'^fuser\.(\d+)', r'fuser.ot_attention.skip.\1', new_k)
            elif _has_base_fuser:
                # Scenario 1: pretrained fuser.X → fuser.base_fuser.X
                new_k = re.sub(r'^fuser\.(\d+)', r'fuser.base_fuser.\1', new_k)
            if new_k != k:
                _n_remapped += 1
            _new_sd[new_k] = v

        if _n_remapped > 0:
            runner.logger.info(f"[load_from] Remapped {_n_remapped} fuser keys")

        missing, unexpected = _model.load_state_dict(_new_sd, strict=False)
        if missing:
            runner.logger.warning(f"[load_from] Missing keys: {missing}")
        if unexpected:
            runner.logger.warning(f"[load_from] Unexpected keys: {unexpected}")
    runner.run(data_loaders, [("train", 1)])
