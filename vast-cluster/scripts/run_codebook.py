#!/usr/bin/env python3
"""DINOv2 and OpenCLIP vectors for one synthetic view. Executed on the GPU box."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import open_clip
import torch
import torchvision.transforms as transforms
from PIL import Image

VIEW = Path("/root/SceneagentX/data/scan/renders/synthetic/view_000.png")
OUT = Path("/root/SceneagentX/data/scan/codebook.json")


def main() -> int:
    if not VIEW.is_file():
        sys.stderr.write(f"missing {VIEW}\n")
        return 2
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(VIEW).convert("RGB")
    dino = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    dino = dino.to(device).eval()
    dino_transform = transforms.Compose(
        [
            transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    tensor = dino_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        dino_vec = dino(tensor).flatten().float().cpu()
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device).eval()
    clip_image = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        clip_vec = model.encode_image(clip_image).flatten().float().cpu()
    vector = torch.cat([dino_vec[:16], clip_vec[:16]]).tolist()
    record = {
        "encoder": "dinov2+openclip",
        "dinov2": True,
        "openclip": True,
        "label": "scan",
        "vector": [round(float(value), 6) for value in vector],
        "view": str(VIEW),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(f"wrote {OUT} dim {len(vector)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
