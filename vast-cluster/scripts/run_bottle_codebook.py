#!/usr/bin/env python3
"""DINOv2 and OpenCLIP physics vector for one object frame.

This is the Block 3 / D object vector, not a LangSplat codebook.
Pass --obj can to write data/can/codebook.json from the cropped can.
The default bottle path is unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import open_clip
import torch
import torchvision.transforms as transforms
from PIL import Image

_REPO = Path(__file__).resolve().parents[2]
if _REPO.is_dir() and str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

FRAMES = Path("/root/SceneagentX/data/bottle/frames/orbit1")
OUT = Path("/root/SceneagentX/data/bottle/codebook.json")


def encode_image(image: Image.Image, device: str) -> list[float]:
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
    return torch.cat([dino_vec[:16], clip_vec[:16]]).tolist()


def _repo_root(root: Path) -> Path:
    candidate = Path(root)
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))
    return candidate


def _crop_from_inverse_mask(view: Path, mask_path: Path) -> Image.Image | None:
    """Background crop: invert the object mask and keep the inverse-mask box."""
    image = Image.open(view).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    rgb = np.asarray(image)
    bg = np.asarray(mask) <= 127
    if not bg.any():
        return None
    ys, xs = np.where(bg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~bg[y0:y1, x0:x1]] = 0
    return Image.fromarray(crop)


def _crop_from_mask(view: Path, mask_path: Path) -> Image.Image:
    image = Image.open(view).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    rgb = np.asarray(image)
    fg = np.asarray(mask) > 127
    if not fg.any():
        return image
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~fg[y0:y1, x0:x1]] = 0
    return Image.fromarray(crop)


def _iou_pair(folder: Path) -> tuple[Path, Path] | None:
    if not folder.is_dir():
        return None
    captures = sorted(
        path
        for path in folder.glob("frame_*.png")
        if not path.name.endswith("_mask.png")
        and not path.name.endswith("_splat.png")
        and not path.name.endswith("_splat_crop.png")
    )
    for view in captures:
        mask = folder / f"{view.stem}_mask.png"
        if mask.is_file():
            return view, mask
    return None


def _codebook_paths(folder: Path, orbit: str) -> tuple[Path, Path]:
    """Orbit 1 keeps the canonical names. Later orbits must not overwrite them."""
    if orbit == "orbit1":
        return folder / "codebook.json", folder / "codebook_gaussians.npz"
    return folder / f"codebook_{orbit}.json", folder / f"codebook_gaussians_{orbit}.npz"


def _copy_onto_gaussians(
    npy: Path,
    vector: list[float],
    out_npz: Path,
    scene_mean: list[float] | None = None,
    index_path: Path | None = None,
    root: Path | None = None,
) -> dict[str, object]:
    inside = np.load(npy)
    if inside.dtype != np.bool_ and inside.dtype != bool:
        inside = inside.astype(bool)
    obj = np.asarray(vector, dtype=np.float32)
    scene_measured = False
    reason = "not assigned; no scene-mean vector was computed"
    assigned = np.full((inside.size, obj.size), np.nan, dtype=np.float32)
    assigned[inside.reshape(-1)] = obj
    if scene_mean is not None:
        if root is not None:
            _repo_root(root)
        from tools.block_d import apply_scene_mean

        result = apply_scene_mean(inside, obj, np.asarray(scene_mean, dtype=np.float64))
        scene_measured = result["scene_mean_measured"] is True
        reason = result["reason"]
        if scene_measured:
            assigned = np.asarray(result["assigned"], dtype=np.float32)
        if index_path is not None:
            index_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(index_path, result["index"])
    out_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_npz,
        object_vector=obj,
        scene_mean=np.asarray(scene_mean, dtype=np.float32) if scene_mean is not None else np.array([], dtype=np.float32),
        inside=inside,
        per_gaussian=assigned,
    )
    return {
        "copied_onto_gaussians": True,
        "copy": "one_object_vector",
        "n_copied": int(inside.sum()),
        "n_gaussians": int(inside.size),
        "assignment": str(npy),
        "file": out_npz.name,
        "scene_mean_measured": scene_measured,
        "outside": "scene mean assigned" if scene_measured else reason,
    }


def _bottle_original() -> int:
    views = sorted(FRAMES.glob("frame_*.png"))
    if not views:
        sys.stderr.write(f"missing frames in {FRAMES}\n")
        return 2
    view = views[len(views) // 2]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = Image.open(view).convert("RGB")
    vector = encode_image(image, device)
    record = {
        "encoder": "dinov2+openclip",
        "dinov2": True,
        "openclip": True,
        "label": "bottle",
        "vector": [round(float(value), 6) for value in vector],
        "view": str(view),
        "copied_onto_gaussians": False,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"BOTTLE_CODEBOOK_OK view {view.name} dim {len(vector)}", flush=True)
    return 0


def _object_codebook(obj: str, orbit: str, root: Path) -> int:
    if obj in {"", ".", "..", "bottle", "vaseline"} or obj != Path(obj).name:
        sys.stderr.write(f"refusing object name {obj}\n")
        return 2
    if orbit != Path(orbit).name or not orbit.startswith("orbit"):
        sys.stderr.write(f"refusing orbit name {orbit}\n")
        return 2
    folder = root / "data" / obj
    pair = _iou_pair(folder / "iou" / orbit)
    if pair is None:
        sys.stderr.write(f"missing cropped iou pair under {folder / 'iou' / orbit}\n")
        return 2
    view, mask = pair
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image = _crop_from_mask(view, mask)
    vector = encode_image(image, device)
    scene_vec: list[float] | None = None
    inverse = _crop_from_inverse_mask(view, mask)
    if inverse is not None:
        scene_vec = encode_image(inverse, device)
    record: dict[str, object] = {
        "encoder": "dinov2+openclip",
        "dinov2": True,
        "openclip": True,
        "label": obj,
        "vector": [round(float(value), 6) for value in vector],
        "view": str(view),
        "mask": str(mask),
        "cropped": True,
        "copied_onto_gaussians": False,
        "learned_prior": False,
        "langsplat_codebook": False,
        "scene_mean_measured": False,
        "note": (
            "DINOv2+OpenCLIP physics object vector from the cropped can. "
            "It is not a LangSplat codebook and not a learned physics prior."
        ),
    }
    npy = folder / "masks" / orbit / "gaussian_in_mask.npy"
    out, npz = _codebook_paths(folder, orbit)
    index_path = (
        folder / "semantics" / "gaussian_feature_idx.npy"
        if orbit == "orbit1"
        else folder / "semantics" / orbit / "gaussian_feature_idx.npy"
    )
    if npy.is_file():
        copy = _copy_onto_gaussians(
            npy,
            record["vector"],
            npz,
            scene_mean=scene_vec,
            index_path=index_path,
            root=root,
        )
        record.update(copy)
        record["learned_prior"] = False
        if copy.get("scene_mean_measured") is True and scene_vec is not None:
            record["scene_mean"] = [round(float(value), 6) for value in scene_vec]
            record["scene_mean_measured"] = True
            record["note"] = (
                f"The DINOv2+OpenCLIP object vector from the cropped {view.name} is copied onto "
                "in-mask Gaussians. A separately encoded inverse-mask crop is the scene mean."
            )
        else:
            record["scene_mean_measured"] = False
            record["note"] = (
                f"The same DINOv2+OpenCLIP vector from the cropped {view.name} is copied onto "
                "in-mask Gaussians. It is not a per-Gaussian embedding, not a LangSplat codebook, "
                "and not a learned physics prior."
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"OBJECT_CODEBOOK_OK obj {obj} orbit {orbit} view {view.name} dim {len(vector)} out {out.name}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="DINOv2+OpenCLIP physics vector")
    parser.add_argument("--obj", default="bottle")
    parser.add_argument("--orbit", default="orbit1")
    parser.add_argument("--root", type=Path, default=Path("/root/SceneagentX"))
    args = parser.parse_args(argv)
    if args.obj == "bottle":
        return _bottle_original()
    return _object_codebook(args.obj, args.orbit, args.root)


if __name__ == "__main__":
    raise SystemExit(main())
