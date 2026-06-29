from typing import Any, Dict

import torch
from mmcv.runner import auto_fp16, force_fp32
from torch import nn
from torch.nn import functional as F

from mmdet3d.models.builder import (
    build_backbone,
    build_fuser,
    build_head,
    build_neck,
    build_vtransform,
)
from mmdet3d.ops import Voxelization, DynamicScatter
from mmdet3d.models import FUSIONMODELS


from .base import Base3DFusionModel

__all__ = ["BEVFusion"]


@FUSIONMODELS.register_module()
class BEVFusion(Base3DFusionModel):
    def __init__(
        self,
        encoders: Dict[str, Any],
        fuser: Dict[str, Any],
        decoder: Dict[str, Any],
        heads: Dict[str, Any],
        **kwargs,
    ) -> None:
        super().__init__()

        self.encoders = nn.ModuleDict()
        if encoders.get("camera") is not None:
            self.encoders["camera"] = nn.ModuleDict(
                {
                    "backbone": build_backbone(encoders["camera"]["backbone"]),
                    "neck": build_neck(encoders["camera"]["neck"]),
                    "vtransform": build_vtransform(encoders["camera"]["vtransform"]),
                }
            )
        if encoders.get("lidar") is not None:
            if encoders["lidar"]["voxelize"].get("max_num_points", -1) > 0:
                voxelize_module = Voxelization(**encoders["lidar"]["voxelize"])
            else:
                voxelize_module = DynamicScatter(**encoders["lidar"]["voxelize"])
            self.encoders["lidar"] = nn.ModuleDict(
                {
                    "voxelize": voxelize_module,
                    "backbone": build_backbone(encoders["lidar"]["backbone"]),
                }
            )
            self.voxelize_reduce = encoders["lidar"].get("voxelize_reduce", True)

        if fuser is not None:
            self._ensure_custom_fuser_registered(fuser)
            self.fuser = build_fuser(fuser)
        else:
            self.fuser = None

        self.decoder = nn.ModuleDict(
            {
                "backbone": build_backbone(decoder["backbone"]),
                "neck": build_neck(decoder["neck"]),
            }
        )
        self.heads = nn.ModuleDict()
        for name in heads:
            if heads[name] is not None:
                self.heads[name] = build_head(heads[name])

        if "loss_scale" in kwargs:
            self.loss_scale = kwargs["loss_scale"]
        else:
            self.loss_scale = dict()
            for name in heads:
                if heads[name] is not None:
                    self.loss_scale[name] = 1.0

        # -------------------------------------------------------------
        # UQ training configuration (optional). Enabled only when the
        # fuser is a UQFuser. A dict like:
        #   {"enable": True, "loss_weight": 0.1}
        # -------------------------------------------------------------
        self.uq_training_cfg = kwargs.get("uq_training", None)
        # Persistent CPU generator for corruption RNG. Not seeded here —
        # keep it independent per process for DDP variety. Caller can
        # seed via manual_seed if reproducibility is needed.
        self._uq_rng = torch.Generator()
        # Stash for one forward: the corruption plan used on this batch.
        self._last_corruption_plan = None
        self.flow_training_cfg = kwargs.get("flow_training", None)

        # -------------------------------------------------------------
        # Freeze mechanism. `freeze` is a list of name prefixes; any
        # parameter whose qualified name starts with any listed prefix
        # gets requires_grad=False. Also disables batchnorm running-
        # stats update in those modules (via switching to eval mode in
        # train()).
        # Example:
        #   freeze:
        #     - "encoders."
        #     - "decoder."
        #     - "heads."
        #     - "fuser.base_fuser."
        # -------------------------------------------------------------
        self._freeze_prefixes = kwargs.get("freeze", None) or []
        if self._freeze_prefixes:
            self._apply_freeze()

        self.init_weights()

    @staticmethod
    def _ensure_custom_fuser_registered(fuser_cfg) -> None:
        """Import custom fusers before FUSERS.build looks them up."""
        if not isinstance(fuser_cfg, dict):
            return
        fuser_type = fuser_cfg.get("type", None)
        if fuser_type in {"UQFuser", "UQFlowFuser"}:
            from mmdet3d.models.reliability.uq_fusion import UQFuser  # noqa: F401
            from mmdet3d.models.reliability.uq_flow_fusion import (  # noqa: F401
                UQFlowFuser,
            )

    def _apply_freeze(self) -> None:
        """Freeze parameters matching any configured prefix.

        Also collects the set of module-paths whose every parameter is
        frozen, so we can hold them in .eval() mode to freeze
        BN running-statistics too.
        """
        frozen_params = 0
        total_params = 0
        for name, p in self.named_parameters():
            total_params += 1
            if any(name.startswith(pre) for pre in self._freeze_prefixes):
                p.requires_grad = False
                frozen_params += 1

        self._frozen_module_prefixes = tuple(self._freeze_prefixes)
        print(
            f"[BEVFusion.freeze] prefixes={self._freeze_prefixes} | "
            f"{frozen_params}/{total_params} parameters frozen"
        )

    def train(self, mode: bool = True):
        """Override so that frozen modules stay in eval() for BN stats."""
        super().train(mode)
        if not getattr(self, "_frozen_module_prefixes", ()):
            return self
        # Put any module whose qualified name starts with a frozen prefix
        # into eval mode, so its BN running stats don't update.
        for mod_name, module in self.named_modules():
            if not mod_name:
                continue  # skip root
            if any(mod_name.startswith(pre.rstrip(".")) for pre in self._frozen_module_prefixes):
                module.eval()
        return self

    def init_weights(self) -> None:
        if "camera" in self.encoders:
            self.encoders["camera"]["backbone"].init_weights()

    def extract_camera_features(
        self,
        x,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        img_metas,
    ) -> torch.Tensor:
        B, N, C, H, W = x.size()
        x = x.view(B * N, C, H, W)

        x = self.encoders["camera"]["backbone"](x)
        x = self.encoders["camera"]["neck"](x)

        if not isinstance(x, torch.Tensor):
            x = x[0]

        BN, C, H, W = x.size()
        x = x.view(B, int(BN / B), C, H, W)

        x = self.encoders["camera"]["vtransform"](
            x,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            img_metas,
        )
        return x

    def extract_lidar_features(self, x) -> torch.Tensor:
        feats, coords, sizes = self.voxelize(x)
        batch_size = coords[-1, 0] + 1
        x = self.encoders["lidar"]["backbone"](feats, coords, batch_size, sizes=sizes)
        return x

    @torch.no_grad()
    @force_fp32()
    def voxelize(self, points):
        feats, coords, sizes = [], [], []
        for k, res in enumerate(points):
            ret = self.encoders["lidar"]["voxelize"](res)
            if len(ret) == 3:
                # hard voxelize
                f, c, n = ret
            else:
                assert len(ret) == 2
                f, c = ret
                n = None
            feats.append(f)
            coords.append(F.pad(c, (1, 0), mode="constant", value=k))
            if n is not None:
                sizes.append(n)

        feats = torch.cat(feats, dim=0)
        coords = torch.cat(coords, dim=0)
        if len(sizes) > 0:
            sizes = torch.cat(sizes, dim=0)
            if self.voxelize_reduce:
                feats = feats.sum(dim=1, keepdim=False) / sizes.type_as(feats).view(
                    -1, 1
                )
                feats = feats.contiguous()

        return feats, coords, sizes

    @auto_fp16(apply_to=("img", "points"))
    def forward(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        if isinstance(img, list):
            raise NotImplementedError

        if (
            self.training
            and self.flow_training_cfg is not None
            and self.flow_training_cfg.get("enable", False)
            and self._fuser_is_flow()
        ):
            return self.forward_flow_train(
                img,
                points,
                camera2ego,
                lidar2ego,
                lidar2camera,
                lidar2image,
                camera_intrinsics,
                camera2lidar,
                img_aug_matrix,
                lidar_aug_matrix,
                metas,
                gt_masks_bev,
                gt_bboxes_3d,
                gt_labels_3d,
                **kwargs,
            )

        # ---- UQ training: apply synthetic corruption to raw inputs ----
        # Only in training, only if enabled, and only if we have a UQFuser.
        self._last_corruption_plan = None
        if (
            self.training
            and self.uq_training_cfg is not None
            and self.uq_training_cfg.get("enable", False)
            and self._fuser_is_uq()
        ):
            # Lazy import to avoid circular ref at module load.
            from mmdet3d.models.reliability import (
                make_corruption_plan,
                apply_corruption,
            )
            B = img.shape[0]
            plans = make_corruption_plan(
                B,
                generator=self._uq_rng,
                clean_prob=self.uq_training_cfg.get("clean_prob", 0.30),
                cam_prob=self.uq_training_cfg.get("cam_prob", 0.45),
            )
            img, points = apply_corruption(img, points, plans, generator=self._uq_rng)
            self._last_corruption_plan = plans
        # ---------------------------------------------------------------

        outputs = self.forward_single(
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
            gt_masks_bev,
            gt_bboxes_3d,
            gt_labels_3d,
            **kwargs,
        )
        return outputs

    def _fuser_is_uq(self) -> bool:
        """Check (without importing at module load) whether fuser is UQFuser."""
        if self.fuser is None:
            return False
        from mmdet3d.models.reliability import UQFuser

        return isinstance(self.fuser, UQFuser)

    def _fuser_is_flow(self) -> bool:
        if self.fuser is None:
            return False
        from mmdet3d.models.reliability import UQFlowFuser

        return isinstance(self.fuser, UQFlowFuser)

    def extract_features(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
    ):
        features = []
        for sensor in (
            self.encoders if self.training else list(self.encoders.keys())[::-1]
        ):
            if sensor == "camera":
                feature = self.extract_camera_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )
            elif sensor == "lidar":
                feature = self.extract_lidar_features(points)
            else:
                raise ValueError(f"unsupported sensor: {sensor}")
            features.append(feature)

        if not self.training:
            features = features[::-1]
        return features

    @auto_fp16(apply_to=("img", "points"))
    def forward_flow_train(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        from mmdet3d.models.reliability import (
            CorruptionPlan,
            apply_corruption,
            make_corruption_plan,
        )

        cfg = self.flow_training_cfg or {}
        enable_fm = bool(cfg.get("enable_fm", True))
        apply_aug = bool(cfg.get("apply_corruption", True))

        self._last_corruption_plan = None
        if apply_aug:
            plans = make_corruption_plan(
                img.shape[0],
                generator=self._uq_rng,
                clean_prob=cfg.get("clean_prob", 0.70),
                cam_prob=cfg.get("cam_prob", 0.20),
            )
            img_deg, points_deg = apply_corruption(
                img, points, plans, generator=self._uq_rng
            )
        else:
            plans = [CorruptionPlan(kind="clean", sub=None) for _ in range(img.shape[0])]
            img_deg, points_deg = img, points
        self._last_corruption_plan = plans
        plan_kinds = [p.kind for p in plans]

        deg_features = self.extract_features(
            img_deg,
            points_deg,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
        )

        clean_features = None
        if enable_fm and any(k != "clean" for k in plan_kinds):
            with torch.no_grad():
                clean_features = self.extract_features(
                    img,
                    points,
                    camera2ego,
                    lidar2ego,
                    lidar2camera,
                    lidar2image,
                    camera_intrinsics,
                    camera2lidar,
                    img_aug_matrix,
                    lidar_aug_matrix,
                    metas,
                )

        x, flow_aux = self.fuser.forward_train_flow(
            deg_features,
            clean_inputs=clean_features,
            plan_kinds=plan_kinds,
            enable_fm=enable_fm,
            num_steps=cfg.get("num_steps", 1),
            fm_loss_type=cfg.get("fm_loss_type", "smooth_l1"),
            smooth_l1_beta=cfg.get("smooth_l1_beta", 0.1),
        )

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        outputs = {}
        det_loss_total = None
        for type, head in self.heads.items():
            if float(self.loss_scale.get(type, 1.0)) == 0.0:
                continue
            if type == "object":
                pred_dict = head(x, metas)
                losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
            elif type == "map":
                losses = head(x, gt_masks_bev)
            else:
                raise ValueError(f"unsupported head: {type}")
            for name, val in losses.items():
                if val.requires_grad:
                    scaled = val * self.loss_scale[type]
                    outputs[f"loss/{type}/{name}"] = scaled
                    det_loss_total = (
                        scaled.detach().mean()
                        if det_loss_total is None
                        else det_loss_total + scaled.detach().mean()
                    )
                else:
                    outputs[f"stats/{type}/{name}"] = val

        if "loss_fm" in flow_aux:
            w = float(cfg.get("loss_weight", 0.05))
            fm_raw = flow_aux.pop("loss_fm")
            fm_weighted = fm_raw * w
            outputs["loss/flow/fm"] = fm_weighted
            outputs["stats/flow/loss_weight"] = torch.tensor(
                w, dtype=torch.float32, device=fm_weighted.device
            )
            outputs["stats/flow/fm_weighted"] = fm_weighted.detach()
            if det_loss_total is not None:
                outputs["stats/flow/det_total"] = det_loss_total
                outputs["stats/flow/fm_to_det"] = (
                    fm_weighted.detach() / det_loss_total.clamp_min(1e-6)
                )
        for name, val in flow_aux.items():
            outputs[name] = val
        return outputs

    @auto_fp16(apply_to=("img", "points"))
    def forward_single(
        self,
        img,
        points,
        camera2ego,
        lidar2ego,
        lidar2camera,
        lidar2image,
        camera_intrinsics,
        camera2lidar,
        img_aug_matrix,
        lidar_aug_matrix,
        metas,
        gt_masks_bev=None,
        gt_bboxes_3d=None,
        gt_labels_3d=None,
        **kwargs,
    ):
        features = self.extract_features(
            img,
            points,
            camera2ego,
            lidar2ego,
            lidar2camera,
            lidar2image,
            camera_intrinsics,
            camera2lidar,
            img_aug_matrix,
            lidar_aug_matrix,
            metas,
        )

        # Keep pre-fuser features for UQ BCE loss in training mode.
        # In training mode the iter order above gave us [cam, lid];
        # in eval we already swapped back but training doesn't reach here
        # with the reversed list — so this slicing is always training-safe.
        uq_pre_fuser_cam = None
        uq_pre_fuser_lid = None
        if self.training and self._fuser_is_uq() and len(features) == 2:
            # features order in training: [cam_bev, lid_bev] — see loop
            # above. extract_camera_features always appended first.
            uq_pre_fuser_cam = features[0]
            uq_pre_fuser_lid = features[1]

        if self.fuser is not None:
            x = self.fuser(features)
        else:
            assert len(features) == 1, features
            x = features[0]

        batch_size = x.shape[0]

        x = self.decoder["backbone"](x)
        x = self.decoder["neck"](x)

        if self.training:
            outputs = {}
            for type, head in self.heads.items():
                if float(self.loss_scale.get(type, 1.0)) == 0.0:
                    continue
                if type == "object":
                    pred_dict = head(x, metas)
                    losses = head.loss(gt_bboxes_3d, gt_labels_3d, pred_dict)
                elif type == "map":
                    losses = head(x, gt_masks_bev)
                else:
                    raise ValueError(f"unsupported head: {type}")
                for name, val in losses.items():
                    if val.requires_grad:
                        outputs[f"loss/{type}/{name}"] = val * self.loss_scale[type]
                    else:
                        outputs[f"stats/{type}/{name}"] = val

            # -------- UQ BCE loss (learned reliability head training) --------
            if (
                self._fuser_is_uq()
                and uq_pre_fuser_cam is not None
                and self.uq_training_cfg is not None
                and self.uq_training_cfg.get("enable", False)
            ):
                from mmdet3d.models.reliability import compute_uq_bce
                # v2: BCE is computed on LOGITS for fp16 stability.
                logit_cam = self.fuser.last_uq_cam_logit
                logit_lid = self.fuser.last_uq_lid_logit
                plans = self._last_corruption_plan
                if logit_cam is not None and plans is not None:
                    plan_kinds = [p.kind for p in plans]
                    uq_out = compute_uq_bce(
                        logit_cam, logit_lid,
                        uq_pre_fuser_cam, uq_pre_fuser_lid,
                        plan_kinds,
                        label_smoothing=self.uq_training_cfg.get("label_smoothing", 0.0),
                        loss_cam_weight=self.uq_training_cfg.get("loss_cam_weight", 1.0),
                        loss_lid_weight=self.uq_training_cfg.get("loss_lid_weight", 1.0),
                        cam_support_keep_frac=self.uq_training_cfg.get("cam_support_keep_frac", 0.5),
                    )
                    w = float(self.uq_training_cfg.get("loss_weight", 0.1))
                    outputs["loss/uq/bce"] = uq_out["loss"] * w
                    outputs["stats/uq/loss_cam"] = uq_out["loss_cam"]
                    outputs["stats/uq/loss_lid"] = uq_out["loss_lid"]
                    outputs["stats/uq/acc_cam"] = uq_out["acc_cam"]
                    outputs["stats/uq/acc_lid"] = uq_out["acc_lid"]
                    outputs["stats/uq/support_cam"] = uq_out["support_cam"]
                    outputs["stats/uq/support_lid"] = uq_out["support_lid"]

                if hasattr(self.fuser, "ot_attention"):
                    attn = self.fuser.ot_attention
                    if getattr(attn, "last_res_skip_ratio", None) is not None:
                        outputs["stats/uq/attn_gate"] = attn.last_attn_gate
                        outputs["stats/uq/target_gate_mean"] = attn.last_target_gate_mean
                        outputs["stats/uq/res_rms"] = attn.last_res_rms
                        outputs["stats/uq/skip_rms"] = attn.last_skip_rms
                        outputs["stats/uq/res_skip_ratio"] = attn.last_res_skip_ratio
            # ----------------------------------------------------------------

            return outputs
        else:
            outputs = [{} for _ in range(batch_size)]
            for type, head in self.heads.items():
                if type == "object":
                    pred_dict = head(x, metas)
                    bboxes = head.get_bboxes(pred_dict, metas)
                    for k, (boxes, scores, labels) in enumerate(bboxes):
                        outputs[k].update(
                            {
                                "boxes_3d": boxes.to("cpu"),
                                "scores_3d": scores.cpu(),
                                "labels_3d": labels.cpu(),
                            }
                        )
                elif type == "map":
                    logits = head(x)
                    for k in range(batch_size):
                        outputs[k].update(
                            {
                                "masks_bev": logits[k].cpu(),
                                "gt_masks_bev": gt_masks_bev[k].cpu(),
                            }
                        )
                else:
                    raise ValueError(f"unsupported head: {type}")
            return outputs
