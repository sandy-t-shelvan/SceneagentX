"""Generate three TRELLIS sisters conditioned on the eight synthetic views."""

from __future__ import annotations

import os

os.environ.setdefault("ATTN_BACKEND", "xformers")
os.environ.setdefault("SPCONV_ALGO", "native")

import sys
from pathlib import Path

from PIL import Image

ROOT = Path("/root/SceneagentX/third_party/TRELLIS")
sys.path.insert(0, str(ROOT))

from trellis.pipelines import TrellisImageTo3DPipeline
from trellis.utils import postprocessing_utils

VIEWS = Path("/root/SceneagentX/data/scan/renders/synthetic")
OUT = Path("/root/trellis_out")


def main() -> None:
    images = [Image.open(VIEWS / f"view_{index:03d}.png").convert("RGB") for index in range(8)]
    if len(images) != 8:
        raise RuntimeError(f"expected 8 synthetic views, found {len(images)}")
    OUT.mkdir(parents=True, exist_ok=True)
    pipeline = TrellisImageTo3DPipeline.from_pretrained("microsoft/TRELLIS-image-large")
    pipeline.cuda()
    for index, seed in enumerate((1, 2, 3), start=1):
        outputs = pipeline.run_multi_image(
            images,
            num_samples=1,
            seed=seed,
            formats=["mesh", "gaussian"],
        )
        glb = postprocessing_utils.to_glb(
            outputs["gaussian"][0],
            outputs["mesh"][0],
            simplify=0.95,
            texture_size=512,
        )
        destination = OUT / f"sister_{index:02d}.glb"
        glb.export(str(destination))
        outputs["gaussian"][0].save_ply(str(OUT / f"sister_{index:02d}.ply"))
        print(f"wrote {destination}", flush=True)


if __name__ == "__main__":
    main()
