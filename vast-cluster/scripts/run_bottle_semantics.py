#!/usr/bin/env python3
"""Label bottle frames with GroundingDINO and SAM 2."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.ops import box_convert

ROOT = Path("/root/SceneagentX/third_party/Grounded-SAM-2")
sys.path.insert(0, str(ROOT))

from grounding_dino.groundingdino.util.inference import load_model, load_image, predict
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

FRAMES = Path("/root/SceneagentX/data/bottle/frames/orbit1")
OUT = Path("/root/SceneagentX/data/bottle/masks")
PROMPT = "bottle."
SAM2_CHECKPOINT = ROOT / "checkpoints" / "sam2.1_hiera_large.pt"
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
DINO_CONFIG = str(ROOT / "grounding_dino/groundingdino/config/GroundingDINO_SwinT_OGC.py")
DINO_CHECKPOINT = ROOT / "gdino_checkpoints" / "groundingdino_swint_ogc.pth"


def main() -> int:
    if not SAM2_CHECKPOINT.is_file() or not DINO_CHECKPOINT.is_file():
        sys.stderr.write("checkpoints are missing\n")
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = build_sam2(SAM2_CONFIG, str(SAM2_CHECKPOINT), device=device)
    predictor = SAM2ImagePredictor(sam)
    grounding = load_model(DINO_CONFIG, str(DINO_CHECKPOINT), device=device)
    OUT.mkdir(parents=True, exist_ok=True)
    views = sorted(FRAMES.glob("frame_*.png"))
    labels = []
    found_n = 0
    for index, view in enumerate(views, start=1):
        image_source, image = load_image(str(view))
        predictor.set_image(image_source)
        boxes, _confidences, names = predict(
            model=grounding,
            image=image,
            caption=PROMPT,
            box_threshold=0.25,
            text_threshold=0.25,
            device=device,
        )
        mask = np.zeros(image_source.shape[:2], dtype=np.uint8)
        if len(boxes) > 0:
            found_n += 1
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
            mask = (masks.any(axis=0).astype(np.uint8)) * 255
            labels.extend(list(names))
        cv2.imwrite(str(OUT / f"{view.stem}_mask.png"), mask)
        if index % 20 == 0 or index == len(views):
            print(f"labeled {index}/{len(views)} detections_so_far {found_n}", flush=True)
    record = {
        "label": "bottle",
        "background": "room",
        "views": "orbit1",
        "segmenter": "grounding_dino_sam2",
        "grounding_dino": True,
        "sam": True,
        "prompt": PROMPT,
        "detections": sorted(set(labels)),
        "n_views": len(views),
        "n_views_with_detection": found_n,
    }
    (OUT / "object.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print("BOTTLE_SEMANTICS_OK", flush=True)
    print(json.dumps(record), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
