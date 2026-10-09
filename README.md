# SceneagentX

Reproduction of SceneAgent (Hollis, Fan, and Yang, Harvard Computational Robotics Group, 2026). The study turns a 3D capture into a simulator scene with semantic features, per-Gaussian physics, parts, articulation, and digital-sister variations.

This checkout implements **B0** through **B4**: the environment probe, the capture checker, and the COLMAP, splatfacto, and alignment wrappers. Later blocks are registered and stop with an error until their GPU tools are added. The numeric B2–B4 gates run on the GPU box. Local tests use saved logs.

## Local checks

From this directory, with Python 3.10+:

```bash
pip install -r requirements-sa.txt
pytest tests -q
python tools/step00_check_frames.py --demo
python -m tools.pipeline --list
python vast-cluster/scripts/search_offer.py
```

`search_offer.py` prints a Vast create command for one on-demand RTX 4090 with 24 cores, 64 GB RAM, 250 GB disk, and driver 580 or newer. It does not rent a machine and it does not store an API key.

On the GPU box, after footage is in `data/marker/raw/`:

```bash
python tools/step01_colmap.py --obj marker --orbit 1
python tools/step02_splatfacto.py --obj marker --orbit 1
python tools/align.py --obj marker --geometry data/marker/align_geometry.json --no-rerun
python -m tools.pipeline --obj marker --through B4
```

`--demo` writes a synthetic orbit under `data/_demo` and runs the B1 checker with a short-orbit frame count. Sharpness, AprilTag, and exposure still use the plan's thresholds.

## GPU box (B0)

The plan's machine is a Linux box with an RTX 4090 or 5090, driver >= 550, and micromamba. Every env pins `torch==2.1.2+cu121`. `env/nerf.yml` also installs COLMAP, nerfstudio 1.1.4, and the gsplat 1.0.0 release that nerfstudio 1.1.4 requires.

```bash
bash setup.sh
micromamba run -n sa pytest tests -q
python tools/check_env.py --obj marker
```

`check_env.py` passes only when nvidia-smi shows a 24 GB-class GPU, all seven envs print `True` for `torch.cuda.is_available()`, `import gsplat` works, and `ns-train --help` lists splatfacto. If torch cannot see CUDA, change the pin in `env/*.yml`. Do not edit `third_party/`.

## Your capture (B1)

### Option A: Video Capture (Original Method)

Tape the marker down. Print a tag36h11 AprilTag at 5 cm and leave it on the table for every orbit. Measure the marker and record the length in metres:

```bash
python -m tools.common.status set --obj marker ground_truth.length_m 0.14
```

Put footage here:

- `data/marker/raw/orbit1.mp4` — cap on
- `data/marker/raw/orbit2.mp4` — cap off, cap beside the body
- `data/marker/raw/orbit3.mp4` — empty table
- `data/marker/raw/stills/` — five side photos

A directory of images named `data/marker/raw/orbit1/` is also accepted. Then:

```bash
python tools/step00_check_frames.py --obj marker
```

The checker extracts frames at 4 fps into `data/marker/frames/orbitN/`, writes `data/marker/renders/step00/orbitN_sheet.png` (every 20th frame), and stores the gate results in `data/marker/status.json`. A real orbit passes when it has 150–250 frames, median Laplacian variance above 100, fewer than 10% of frames below 50, the tag in at least 80% of frames, and luminance standard deviation under 5% of the mean.

B0 has to pass before `python -m tools.pipeline --obj marker --through B1` will check footage. On a machine without the GPU stack, run the frame checker directly.

### Option B: Polycam Scan (Alternative Method)

If video capture fails due to sharpness, lighting, or AprilTag detection issues, use a Polycam scan instead:

```bash
# 1. Scan object with Polycam app (iPhone/iPad with LiDAR or any phone with Photo mode)
# 2. Export as COLMAP or "All Data" format
# 3. Place in data/marker/polycam/
# 4. Measure marker manually
python -m tools.common.status set --obj marker ground_truth.length_m 0.14

# 5. Import Polycam data
python tools/import_polycam.py --obj marker --scan data/marker/polycam/ --orbit 1 --scale 0.14

# 6. Continue with splatfacto
python tools/step02_splatfacto.py --obj marker --orbit 1
python tools/align.py --obj marker --manual-scale 0.14 --no-rerun
python -m tools.pipeline --obj marker --from B5
```

See [POLYCAM_WORKFLOW.md](POLYCAM_WORKFLOW.md) for detailed instructions.

## What you still supply

Phase 1 does not download model weights. Before the later blocks, clone the upstream projects into `third_party/` and follow each project's own weight instructions:

| Block | Env | Upstream |
| --- | --- | --- |
| B2–B4 | `nerf` | nerfstudio and gsplat, installed by `env/nerf.yml` |
| B5 | `gsam2` | https://github.com/IDEA-Research/Grounded-SAM-2 |
| B8 | `2dgs` | https://github.com/hbb1/2d-gaussian-splatting |
| B9 | `partfield` | https://github.com/nv-tlabs/PartField |
| B11, B15 | `physgauss` | https://github.com/XPandora/PhysGaussian and Genesis |
| B14 | `trellis` | https://github.com/microsoft/TRELLIS |

B7 also needs DINOv2 and OpenCLIP weights. B10 needs a vision-language model endpoint of your choice. No API keys are stored in this repo.
