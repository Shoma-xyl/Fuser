"""Create a nuScenes train split that excludes night/rain scenes.

This is a lightweight split generator for clean-to-adverse experiments:

    train: non-night/non-rain scenes from nuscenes_infos_train.pkl
    val:   unchanged full nuscenes_infos_val.pkl

nuScenes does not provide a strict "clear/day" boolean in the infos pkl.
We therefore use scene.description from the NuScenes SDK and keep samples
whose scene description does not match exclusion keywords. By default those
keywords are "night", "rain", and "raining".

Usage:
    python tools/create_nuscenes_clear_day_split.py \
        --nuscenes-root data/nuscenes \
        --version v1.0-trainval
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import mmcv


DEFAULT_EXCLUDE_KEYWORDS = ("night", "rain", "raining")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nuscenes-root",
        default="data/nuscenes",
        help="nuScenes dataroot containing v1.0-trainval and infos pkl files.",
    )
    parser.add_argument(
        "--version",
        default="v1.0-trainval",
        help="NuScenes SDK version used for scene descriptions.",
    )
    parser.add_argument(
        "--in-file",
        default=None,
        help="Input train infos pkl. Defaults to <root>/nuscenes_infos_train.pkl.",
    )
    parser.add_argument(
        "--out-file",
        default=None,
        help="Output filtered pkl. Defaults to <root>/nuscenes_infos_train_clear_day.pkl.",
    )
    parser.add_argument(
        "--exclude-keywords",
        nargs="+",
        default=list(DEFAULT_EXCLUDE_KEYWORDS),
        help="Lowercase scene-description keywords to exclude.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print split statistics without writing the output pkl.",
    )
    return parser.parse_args()


def matches_any(description: str, keywords: Iterable[str]) -> str | None:
    text = description.lower()
    for keyword in keywords:
        keyword = keyword.lower()
        if keyword and keyword in text:
            return keyword
    return None


def build_scene_lookup(nuscenes_root: Path, version: str) -> tuple[dict[str, str], dict[str, str]]:
    from nuscenes.nuscenes import NuScenes

    nusc = NuScenes(version=version, dataroot=str(nuscenes_root), verbose=False)
    token_to_desc: dict[str, str] = {}
    token_to_scene: dict[str, str] = {}
    for sample in nusc.sample:
        scene = nusc.get("scene", sample["scene_token"])
        token_to_desc[sample["token"]] = scene.get("description", "")
        token_to_scene[sample["token"]] = sample["scene_token"]
    return token_to_desc, token_to_scene


def main() -> None:
    args = parse_args()
    root = Path(args.nuscenes_root).expanduser()
    in_file = Path(args.in_file).expanduser() if args.in_file else root / "nuscenes_infos_train.pkl"
    out_file = (
        Path(args.out_file).expanduser()
        if args.out_file
        else root / "nuscenes_infos_train_clear_day.pkl"
    )

    data = mmcv.load(str(in_file))
    infos = data["infos"]
    token_to_desc, token_to_scene = build_scene_lookup(root, args.version)

    kept_infos = []
    kept_scenes = set()
    excluded_scenes = set()
    reason_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)

    for info in infos:
        token = info.get("token")
        desc = token_to_desc.get(token, "")
        scene_token = token_to_scene.get(token, "")
        reason = matches_any(desc, args.exclude_keywords)
        if reason is None:
            kept_infos.append(info)
            if scene_token:
                kept_scenes.add(scene_token)
            continue

        reason_counts[reason] += 1
        if scene_token:
            excluded_scenes.add(scene_token)
        if len(examples[reason]) < 5:
            examples[reason].append(desc)

    ratio = len(kept_infos) / max(1, len(infos))
    print(f"[split] input:  {in_file}")
    print(f"[split] output: {out_file}")
    print(f"[split] kept samples: {len(kept_infos)}/{len(infos)} ({ratio:.2%})")
    print(f"[split] kept scenes:  {len(kept_scenes)}")
    print(f"[split] excluded scenes: {len(excluded_scenes)}")
    if reason_counts:
        print("[split] excluded sample counts by keyword:")
        for key, count in reason_counts.most_common():
            print(f"  {key}: {count}")
            for desc in examples[key]:
                print(f"    example: {desc}")

    out_data = dict(data)
    out_data["infos"] = kept_infos
    metadata = dict(out_data.get("metadata", {}))
    metadata["clear_day_split"] = {
        "source": str(in_file),
        "version": args.version,
        "exclude_keywords": list(args.exclude_keywords),
        "kept_samples": len(kept_infos),
        "total_samples": len(infos),
        "kept_ratio": ratio,
    }
    out_data["metadata"] = metadata

    if args.dry_run:
        print("[split] dry run: not writing output pkl")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)
    mmcv.dump(out_data, str(out_file))
    print("[split] wrote filtered train infos")


if __name__ == "__main__":
    main()
