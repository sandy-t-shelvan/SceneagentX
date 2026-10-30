#!/usr/bin/env python3
"""Label one object's COLMAP frames with GroundingDINO and SAM 2.

The laptop task spec names the one target. This script detects that name
and does not OR other labels into the mask. It does not parse the user's
sentence and it does not call a vision API.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

GSAM_ROOT = Path("/root/SceneagentX/third_party/Grounded-SAM-2")
sys.path.insert(0, str(GSAM_ROOT))

from grounding_dino.groundingdino.util.inference import load_image, load_model, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

SAM2_CHECKPOINT = GSAM_ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DINO_CONFIG = str(GSAM_ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_CHECKPOINT = GSAM_ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}


def reject_object(obj: str) -> None:
    if obj != Path(obj).name or obj in {"", ".", ".."}:
        raise SystemExit(f"refusing object name {obj}")


def colmap_images(colmap: Path) -> list[Path]:
    images = colmap / "images"
    if images.is_dir():
        found = sorted(path for path in images.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
        if found:
            return found
    return sorted(
        path
        for path in colmap.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES and "sparse" not in path.parts
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="GroundingDINO + SAM 2 masks for one object")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--target", default="", help="Sister target. Other detector names are not unioned into the mask.")
    parser.add_argument("--box-threshold", type=float, default=0.25)
    parser.add_argument("--orbit", default="orbit1", help="orbit name, e.g. orbit2; picks colmap/<orbit> and masks/<orbit>")
    parser.add_argument("--root", type=Path, default=Path("/root/SceneagentX"))
    args = parser.parse_args(argv)
    reject_object(args.obj)
    if args.orbit != Path(args.orbit).name or not args.orbit.startswith("orbit"):
        raise SystemExit(f"refusing orbit name {args.orbit}")
    if not SAM2_CHECKPOINT.is_file() or not DINO_CHECKPOINT.is_file():
        sys.stderr.write("checkpoints are missing\n")
        return 2
    colmap = args.root / "data" / args.obj / "colmap" / args.orbit
    views = colmap_images(colmap)
    if not views:
        sys.stderr.write(f"no images under {colmap}\n")
        return 1
    out = args.root / "data" / args.obj / "masks" / args.orbit
    out.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        sys.stderr.write("cuda is not available\n")
        return 2
    sam = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(sam)
    grounding = load_model(DINO_CONFIG, str(DINO_CHECKPOINT), device=device)
    repo = args.root if (args.root / "tools" / "block_b_job.py").is_file() else Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from tools.block_b_job import union_target_mask

    labels: list[str] = []
    found_n = 0
    largest_name = ""
    largest_pixels = -1
    for index, view in enumerate(views, start=1):
        image_source, image = load_image(str(view))
        predictor.set_image(image_source)
        boxes, _confidences, names = predict(
            model=grounding,
            image=image,
            caption=args.prompt,
            box_threshold=args.box_threshold,
            text_threshold=0.25,
            device=device,
        )
        mask = np.zeros(image_source.shape[:2], dtype=np.uint8)
        if len(boxes) > 0:
            height, width, _ = image_source.shape
            scaled = boxes * torch.tensor([width, height, width, height])
            xyxy = box_convert(boxes=scaled, in_fmt="cxcywh", out_fmt="xyxy").numpy()
            masks, _, _ = predictor.predict(
                point_coords=None,
                point_labels=None,
                box=xyxy,
                multimask_output=False,
            )
            if masks.ndim == 4:
                masks = masks.squeeze(1)
            name_list = [str(item) for item in list(names)]
            count = min(len(name_list), int(masks.shape[0]))
            merged, kept = union_target_mask(masks[:count], name_list[:count], args.target)
            if kept and bool(np.asarray(merged).any()):
                found_n += 1
                mask = (np.asarray(merged).astype(np.uint8)) * 255
                labels.extend(kept)
        cv2.imwrite(str(out / f"{view.stem}_mask.png"), mask)
        pixels = int((mask > 127).sum())
        if pixels > largest_pixels or (pixels == largest_pixels and (not largest_name or view.name < largest_name)):
            largest_pixels = pixels
            largest_name = view.name
        if index % 20 == 0 or index == len(views):
            print(f"labeled {index}/{len(views)} detections_so_far {found_n}", flush=True)
    record = {
        "label": args.obj,
        "views": args.orbit,
        "segmenter": "grounding_dino_sam2",
        "grounding_dino": True,
        "sam": True,
        "prompt": args.prompt,
        "target": args.target,
        "detections": sorted(set(labels)),
        "n_views": len(views),
        "n_views_with_detection": found_n,
        "largest_frame": largest_name,
        "largest_foreground_pixels": largest_pixels,
    }
    (out / "object.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("OBJECT_SEMANTICS_OK", flush=True)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
