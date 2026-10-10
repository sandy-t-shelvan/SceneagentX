#!/usr/bin/env python3
"""GroundingDINO + SAM on the synthetic orbit. Runs on the GPU box."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoModelForZeroShotObjectDetection,
    AutoProcessor,
    SamModel,
    SamProcessor,
)

VIEWS = Path("/root/SceneagentX/data/scan/renders/synthetic")
OUT = Path("/root/SceneagentX/data/scan/masks")
PROMPTS = ("camera.", "object.")
GDINO = "IDEA-Research/grounding-dino-tiny"
SAM = "facebook/sam-vit-base"


def detect(processor, model, image: Image.Image, prompt: str, device: str):
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    try:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
    except TypeError:
        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.25,
            text_threshold=0.25,
            target_sizes=[image.size[::-1]],
        )[0]
    return results


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    gdino_processor = AutoProcessor.from_pretrained(GDINO)
    gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO).to(device).eval()
    sam_processor = SamProcessor.from_pretrained(SAM)
    sam = SamModel.from_pretrained(SAM).to(device).eval()
    OUT.mkdir(parents=True, exist_ok=True)
    found_any = False
    used_prompt = ""
    labels: list[str] = []
    views = sorted(VIEWS.glob("view_*.png"))
    for view in views:
        image = Image.open(view).convert("RGB")
        results = None
        for prompt in PROMPTS:
            results = detect(gdino_processor, gdino, image, prompt, device)
            if len(results["boxes"]):
                used_prompt = prompt
                break
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        boxes = results["boxes"] if results is not None else []
        if len(boxes):
            found_any = True
            labels.extend(str(label) for label in results["labels"])
            sam_inputs = sam_processor(
                image,
                input_boxes=[[box.tolist() for box in boxes]],
                return_tensors="pt",
            ).to(device)
            with torch.no_grad():
                sam_outputs = sam(**sam_inputs)
            processed = sam_processor.image_processor.post_process_masks(
                sam_outputs.pred_masks.cpu(),
                sam_inputs["original_sizes"].cpu(),
                sam_inputs["reshaped_input_sizes"].cpu(),
            )[0]
            combined = processed.any(dim=0).any(dim=0).numpy()
            mask = combined.astype(np.uint8) * 255
        Image.fromarray(mask).save(OUT / f"{view.stem}_mask.png")
    record = {
        "label": "camera",
        "background": "none",
        "views": "synthetic",
        "segmenter": "grounding_dino+sam",
        "grounding_dino": found_any,
        "sam": found_any,
        "prompt": used_prompt,
        "detections": sorted(set(labels)),
        "n_views": len(views),
        "models": {"grounding_dino": GDINO, "sam": SAM},
    }
    (OUT / "object.json").write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(record) + "\n")
    return 0 if found_any else 1


if __name__ == "__main__":
    raise SystemExit(main())
