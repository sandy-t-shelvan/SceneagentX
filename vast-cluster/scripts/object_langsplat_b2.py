#!/usr/bin/env python3
"""Block B B2: CLIP at whole / part / subpart, then a LangSplat codebook.

Uses existing B1 SAM masks. Writes data/<obj>/semantics/features.codebook.npz.
Does not touch codebook.json (physics vector).
Does not retrain splatfacto. Does not lift a feature field (that is B3).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
ORBITS = ("orbit1", "orbit2", "orbit3")
LEVEL_WHOLE, LEVEL_PART, LEVEL_SUBPART = 0, 1, 2
LEVEL_NAMES = ("whole", "part", "subpart")
MIN_PIXELS = 64
CLIP_DIM = 512
LATENT_DIM = 3
N_CODES = 256
AE_EPOCHS = 200
AE_BATCH = 64
PART_GRID = 2
SUBPART_GRID = 4


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


def _mlp(dims: list[int], in_dim: int) -> nn.Sequential:
    """Linear stack. The last layer stays linear so the latent and CLIP vector can be negative."""
    layers: list[nn.Module] = []
    for index, hidden in enumerate(dims):
        layers.append(nn.Linear(in_dim, hidden, bias=False))
        if index != len(dims) - 1:
            layers.append(nn.ReLU())
        in_dim = hidden
    return nn.Sequential(*layers)


class LangSplatAutoencoder(nn.Module):
    """512-d CLIP to 3-d latent. Hidden sizes match LangSplat's autoencoder."""

    def __init__(self) -> None:
        super().__init__()
        encoder_dims = [256, 128, 64, 32, LATENT_DIM]
        decoder_dims = [16, 32, 64, 128, 256, 256, CLIP_DIM]
        self.encoder = _mlp(encoder_dims, CLIP_DIM)
        self.decoder = _mlp(decoder_dims, LATENT_DIM)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        return self.decoder(z), z


def _load_rgb_mask(view: Path, mask_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    if not mask_path.is_file():
        return None
    image = Image.open(view).convert("RGB")
    mask = Image.open(mask_path).convert("L")
    if mask.size != image.size:
        mask = mask.resize(image.size, Image.NEAREST)
    full = np.asarray(mask) > 127
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.block_b_job import mask_covers_frame

    if mask_covers_frame(int(full.sum()), int(full.size)):
        print(
            f"B2_DROP_FRAME {view.name} covers {int(full.sum())}/{int(full.size)}",
            flush=True,
        )
        return None
    # CLIP uses 224px. Downscale the frame before tiling so 4K orbits do not stall.
    if max(image.size) > 1024:
        image.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
        mask = mask.resize(image.size, Image.NEAREST)
    rgb = np.asarray(image)
    fg = np.asarray(mask) > 127
    if int(fg.sum()) < MIN_PIXELS:
        return None
    return rgb, fg.astype(np.uint8)


def _crop(rgb: np.ndarray, fg: np.ndarray) -> Image.Image | None:
    if not fg.any():
        return None
    ys, xs = np.where(fg)
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    if (y1 - y0) < 2 or (x1 - x0) < 2:
        return None
    crop = rgb[y0:y1, x0:x1].copy()
    crop[~fg[y0:y1, x0:x1].astype(bool)] = 0
    image = Image.fromarray(crop)
    # CLIP resizes to 224; keep a short side so 4K masks do not fill RAM.
    if max(image.size) > 512:
        image.thumbnail((512, 512), Image.Resampling.BICUBIC)
    return image


def _grid_tiles(fg: np.ndarray, grid: int) -> list[np.ndarray]:
    ys, xs = np.where(fg)
    if ys.size == 0:
        return []
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    h = y1 - y0
    w = x1 - x0
    tiles: list[np.ndarray] = []
    for i in range(grid):
        for j in range(grid):
            ty0 = y0 + i * h // grid
            ty1 = y0 + (i + 1) * h // grid
            tx0 = x0 + j * w // grid
            tx1 = x0 + (j + 1) * w // grid
            tile = np.zeros_like(fg)
            tile[ty0:ty1, tx0:tx1] = fg[ty0:ty1, tx0:tx1]
            if int(tile.sum()) >= MIN_PIXELS:
                tiles.append(tile)
    return tiles


def _collect_crops(folder: Path) -> tuple[list[Image.Image], list[int], list[str], dict[str, dict]]:
    crops: list[Image.Image] = []
    levels: list[int] = []
    orbits: list[str] = []
    per_orbit: dict[str, dict] = {}
    for orbit in ORBITS:
        views = colmap_images(folder / "colmap" / orbit)
        mask_dir = folder / "masks" / orbit
        counts = {"views": len(views), "with_mask": 0, "whole": 0, "part": 0, "subpart": 0, "empty": 0}
        for index, view in enumerate(views, start=1):
            loaded = _load_rgb_mask(view, mask_dir / f"{view.stem}_mask.png")
            if loaded is None:
                counts["empty"] += 1
                continue
            rgb, fg = loaded
            counts["with_mask"] += 1
            whole = _crop(rgb, fg)
            if whole is None:
                counts["empty"] += 1
                continue
            crops.append(whole)
            levels.append(LEVEL_WHOLE)
            orbits.append(orbit)
            counts["whole"] += 1
            for tile in _grid_tiles(fg, PART_GRID):
                part = _crop(rgb, tile)
                if part is None:
                    continue
                crops.append(part)
                levels.append(LEVEL_PART)
                orbits.append(orbit)
                counts["part"] += 1
            for tile in _grid_tiles(fg, SUBPART_GRID):
                sub = _crop(rgb, tile)
                if sub is None:
                    continue
                crops.append(sub)
                levels.append(LEVEL_SUBPART)
                orbits.append(orbit)
                counts["subpart"] += 1
            if index % 40 == 0 or index == len(views):
                print(f"B2_CROP_PROGRESS {orbit} {index}/{len(views)} crops={len(crops)}", flush=True)
        per_orbit[orbit] = counts
        print(f"B2_CROPS {orbit} {json.dumps(counts)}", flush=True)
    return crops, levels, orbits, per_orbit


def _load_clip(device: str):
    import open_clip

    last_error = None
    for name, pretrained in (("ViT-B-16", "laion2b_s34b_b88k"), ("ViT-B-16", "openai"), ("ViT-B-32", "openai")):
        try:
            model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
            model = model.to(device).eval()
            return model, preprocess, f"openclip_{name}_{pretrained}"
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            print(f"B2_CLIP_TRY {name} {pretrained} failed {exc}", flush=True)
    raise RuntimeError(f"open_clip model load failed: {last_error}")


@torch.no_grad()
def _encode_clip(model, preprocess, crops: list[Image.Image], device: str) -> np.ndarray:
    out = []
    batch = 16
    for start in range(0, len(crops), batch):
        chunk = crops[start : start + batch]
        tensor = torch.stack([preprocess(image) for image in chunk], dim=0).to(device)
        feats = model.encode_image(tensor).float()
        feats = F.normalize(feats, dim=-1)
        out.append(feats.cpu().numpy())
        if start % 256 == 0 or start + batch >= len(crops):
            print(f"B2_CLIP {min(start + batch, len(crops))}/{len(crops)}", flush=True)
    features = np.concatenate(out, axis=0).astype(np.float32)
    if features.shape[1] != CLIP_DIM:
        raise RuntimeError(f"CLIP dim {features.shape[1]} is not {CLIP_DIM}")
    return features


def _train_autoencoder(features: np.ndarray, device: str) -> tuple[LangSplatAutoencoder, np.ndarray, float]:
    x = torch.from_numpy(features).to(device)
    ae = LangSplatAutoencoder().to(device)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    n = x.shape[0]
    last = 0.0
    for epoch in range(AE_EPOCHS):
        ae.train()
        perm = torch.randperm(n, device=device)
        losses = []
        for start in range(0, n, AE_BATCH):
            batch = x[perm[start : start + AE_BATCH]]
            if batch.shape[0] < 2:
                continue
            recon, _latent = ae(batch)
            loss = (1.0 - F.cosine_similarity(recon, batch, dim=-1)).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))
        last = float(np.mean(losses)) if losses else 0.0
        if epoch == 0 or (epoch + 1) % 50 == 0 or epoch + 1 == AE_EPOCHS:
            print(f"B2_AE epoch {epoch + 1}/{AE_EPOCHS} loss {last:.4f}", flush=True)
    ae.eval()
    with torch.no_grad():
        _recon, latents = ae(x)
    return ae, latents.cpu().numpy().astype(np.float32), last


def _kmeans(latents: np.ndarray, k: int, iters: int = 40) -> tuple[np.ndarray, np.ndarray]:
    n = latents.shape[0]
    k = max(1, min(k, n))
    rng = np.random.default_rng(0)
    centers = latents[rng.choice(n, size=k, replace=False)].copy()
    assign = np.zeros(n, dtype=np.int32)
    for _ in range(iters):
        dist = ((latents[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        assign = dist.argmin(axis=1).astype(np.int32)
        for i in range(k):
            members = latents[assign == i]
            if members.size:
                centers[i] = members.mean(axis=0)
    return centers.astype(np.float32), assign


def _backup_physics_npz(npz: Path) -> str | None:
    if not npz.is_file():
        return None
    try:
        with np.load(npz) as book:
            keys = set(book.files)
            if "codebook" in keys and "clip_features" in keys and "levels" in keys:
                return None
            if "object_vector" not in keys:
                return None
    except Exception:  # noqa: BLE001
        return None
    dest = npz.with_name("features.physics_vector.npz")
    shutil.copy2(npz, dest)
    return dest.name


def _write_fail(folder: Path, reason: str, extra: dict | None = None) -> int:
    record = {
        "block": "B",
        "sub_step": "B2",
        "object": folder.name,
        "does": "CLIP on each mask at whole / part / subpart (LangSplat), then a low-d codebook",
        "state": "fail",
        "achieved": False,
        "reason": reason,
        "langsplat": {
            "state": "fail",
            "three_level_clip": "fail",
            "autoencoder": "unstarted",
            "kmeans_codebook": "unstarted",
            "langsplat_codebook": False,
        },
        "milestone": "no M2 number from this step",
    }
    if extra:
        record.update(extra)
    _dump(folder / "ingest" / "block_b_b2.json", record)
    _dump(
        folder / "semantics" / "langsplat_codebook.json",
        {
            "langsplat_codebook": False,
            "state": "fail",
            "achieved": False,
            "reason": reason,
            "physics_vector_is_not_this_codebook": True,
            "file": f"data/{folder.name}/codebook.json",
        },
    )
    print(f"B2_FAIL {reason}", flush=True)
    return 1


def _dump(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LangSplat CLIP-at-3-levels codebook from B1 masks")
    parser.add_argument("--obj", required=True)
    parser.add_argument("--root", type=Path, default=Path("/root/SceneagentX"))
    args = parser.parse_args(argv)
    reject_object(args.obj)
    folder = args.root / "data" / args.obj
    if not torch.cuda.is_available():
        return _write_fail(folder, "cuda is not available")
    device = "cuda"
    root = Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from tools.block_b_job import OPEN_CLIP_MISSING, codebook_collapsed

    try:
        import open_clip  # noqa: F401
    except ModuleNotFoundError as exc:
        if getattr(exc, "name", None) != "open_clip":
            raise
        return _write_fail(folder, OPEN_CLIP_MISSING)
    physics = folder / "codebook.json"
    physics_present = physics.is_file()
    if not physics_present:
        print(
            f"B2_PHYSICS_ABSENT data/{args.obj}/codebook.json was not created and was not renamed",
            flush=True,
        )

    crops, levels, orbits, per_orbit = _collect_crops(folder)
    n_whole = int(sum(1 for item in levels if item == LEVEL_WHOLE))
    n_part = int(sum(1 for item in levels if item == LEVEL_PART))
    n_sub = int(sum(1 for item in levels if item == LEVEL_SUBPART))
    if n_whole < 1 or n_part < 1 or n_sub < 1:
        return _write_fail(
            folder,
            "need CLIP crops at whole, part, and subpart from existing B1 masks",
            {"per_orbit": per_orbit, "n_crops": len(crops)},
        )
    print(f"B2_N_CROPS {len(crops)} whole={n_whole} part={n_part} subpart={n_sub}", flush=True)

    try:
        model, preprocess, clip_name = _load_clip(device)
        clip_features = _encode_clip(model, preprocess, crops, device)
    except Exception as exc:  # noqa: BLE001
        return _write_fail(folder, f"CLIP extract failed: {exc}", {"per_orbit": per_orbit})

    try:
        ae, latents, ae_loss = _train_autoencoder(clip_features, device)
        if codebook_collapsed(float(np.max(np.abs(latents))), ae_loss):
            return _write_fail(
                folder,
                f"autoencoder collapsed: ae_loss {ae_loss}",
                {
                    "per_orbit": per_orbit,
                    "langsplat": {
                        "state": "fail",
                        "three_level_clip": "measured",
                        "autoencoder": "fail",
                        "kmeans_codebook": "unstarted",
                        "langsplat_codebook": False,
                    },
                },
            )
        k = min(N_CODES, int(latents.shape[0]))
        centers, assign = _kmeans(latents, k)
        with torch.no_grad():
            decoded = ae.decoder(torch.from_numpy(centers).to(device)).cpu().numpy().astype(np.float32)
    except Exception as exc:  # noqa: BLE001
        return _write_fail(
            folder,
            f"autoencoder or k-means failed: {exc}",
            {
                "per_orbit": per_orbit,
                "langsplat": {
                    "state": "fail",
                    "three_level_clip": "measured",
                    "autoencoder": "fail",
                    "kmeans_codebook": "unstarted",
                    "langsplat_codebook": False,
                },
            },
        )

    semantics = folder / "semantics"
    semantics.mkdir(parents=True, exist_ok=True)
    npz = semantics / "features.codebook.npz"
    backup = _backup_physics_npz(npz)
    np.savez(
        npz,
        codebook=centers,
        clip_codebook=decoded,
        latents=latents,
        clip_features=clip_features,
        levels=np.asarray(levels, dtype=np.int16),
        assignments=assign,
        orbits=np.asarray(orbits),
    )
    ae_path = semantics / "langsplat_autoencoder.pt"
    torch.save({"encoder": ae.encoder.state_dict(), "decoder": ae.decoder.state_dict(), "latent_dim": LATENT_DIM}, ae_path)

    counts = {
        "n_features": int(clip_features.shape[0]),
        "n_whole": n_whole,
        "n_part": n_part,
        "n_subpart": n_sub,
        "n_codes": int(centers.shape[0]),
        "latent_dim": LATENT_DIM,
        "clip_dim": int(clip_features.shape[1]),
        "ae_loss": ae_loss,
        "clip_encoder": clip_name,
        "per_orbit": per_orbit,
        "physics_npz_backup": backup,
    }
    sidecar = {
        "langsplat_codebook": True,
        "state": "measured",
        "achieved": False,
        "encoder": clip_name,
        "levels": list(LEVEL_NAMES),
        "three_level_clip": "measured",
        "autoencoder": "trained",
        "kmeans_codebook": "trained",
        "latent_dim": LATENT_DIM,
        "n_codes": int(centers.shape[0]),
        "n_features": int(clip_features.shape[0]),
        "n_whole": n_whole,
        "n_part": n_part,
        "n_subpart": n_sub,
        "ae_cosine_loss": ae_loss,
        "physics_vector_is_not_this_codebook": True,
        "physics_vector_file": f"data/{args.obj}/codebook.json" if physics_present else None,
        "file": f"data/{args.obj}/semantics/features.codebook.npz",
        "autoencoder_file": f"data/{args.obj}/semantics/langsplat_autoencoder.pt",
        "learned_prior": False,
        "milestone": "no M2 number from this step",
        "note": "CLIP features come from existing B1 SAM masks at whole/part/subpart. This is not the DINOv2+OpenCLIP physics vector.",
    }
    sidecar.update({"per_orbit": per_orbit})
    _dump(semantics / "langsplat_codebook.json", sidecar)

    physics_rec = json.loads(physics.read_text(encoding="utf-8")) if physics.is_file() else {}
    vector = physics_rec.get("vector")
    b2 = {
        "block": "B",
        "sub_step": "B2",
        "object": args.obj,
        "does": "CLIP on each mask at whole / part / subpart (LangSplat), then a low-d codebook",
        "state": "measured",
        "achieved": False,
        "reason": "LangSplat codebook is written from three-level CLIP; M2 is not this step; achieved stays false",
        "langsplat": {
            "state": "measured",
            "three_level_clip": "measured",
            "autoencoder": "trained",
            "kmeans_codebook": "trained",
            "langsplat_codebook": True,
            "encoder": clip_name,
            "levels": list(LEVEL_NAMES),
            "n_features": int(clip_features.shape[0]),
            "n_whole": n_whole,
            "n_part": n_part,
            "n_subpart": n_sub,
            "n_codes": int(centers.shape[0]),
            "latent_dim": LATENT_DIM,
            "clip_dim": CLIP_DIM,
            "ae_cosine_loss": ae_loss,
            "file": f"data/{args.obj}/semantics/features.codebook.npz",
            "sidecar": f"data/{args.obj}/semantics/langsplat_codebook.json",
            "physics_npz_backup": backup,
        },
        "object_vector": {
            "state": "measured" if physics_present else "missing",
            "achieved": False,
            "encoder": physics_rec.get("encoder", "dinov2+openclip") if physics_present else None,
            "dinov2": physics_rec.get("dinov2") is True,
            "openclip": physics_rec.get("openclip") is True,
            "learned_prior": False,
            "langsplat": False,
            "dim": len(vector) if isinstance(vector, list) else None,
            "n_copied": physics_rec.get("n_copied"),
            "n_gaussians": physics_rec.get("n_gaussians"),
            "view": physics_rec.get("view"),
            "mask": physics_rec.get("mask"),
            "cropped": physics_rec.get("cropped"),
            "file": f"data/{args.obj}/codebook.json" if physics_present else None,
            "renamed": False,
            "note": (
                "one DINOv2+OpenCLIP object vector on the object crop; it is an input to physics, not a Block B pass"
                if physics_present
                else "physics vector file is absent; it was not created and was not renamed"
            ),
        },
        "scene_mean": {
            "measured": False,
            "reason": "one mixed crop is not a scene mean; scene mean is not a separate measurement",
        },
        "milestone": "no M2 number from this step",
        "note": "DINOv2 and OpenCLIP physics vector were not rerun. codebook.json was not renamed.",
        "orbits": [],
    }
    for orbit in ORBITS:
        codebook_name = "codebook.json" if orbit == "orbit1" else f"codebook_{orbit}.json"
        orbit_book = folder / codebook_name
        rec = json.loads(orbit_book.read_text(encoding="utf-8")) if orbit_book.is_file() else {}
        vec = rec.get("vector")
        b2["orbits"].append(
            {
                "orbit": orbit,
                "state": "measured" if orbit_book.is_file() else "missing",
                "achieved": False,
                "langsplat_crops": per_orbit.get(orbit),
                "encoder": rec.get("encoder", "dinov2+openclip"),
                "dim": len(vec) if isinstance(vec, list) else None,
                "n_copied": rec.get("n_copied"),
                "n_gaussians": rec.get("n_gaussians"),
                "file": f"data/{args.obj}/{codebook_name}",
                "learned_prior": False,
                "langsplat": False,
                "note": "orbit physics vector unchanged; LangSplat codebook is shared in semantics/features.codebook.npz",
            }
        )
    _dump(folder / "ingest" / "block_b_b2.json", b2)
    print("B2_LANGSPLAT_OK", json.dumps(counts), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
