# Vast AI notes

Reference for the SceneAgent GPU machines. Look up the live host before every SSH. Do not reuse an address from this file if `vastai show instances` disagrees.

```powershell
.\.venv\Scripts\vastai.exe show instances --raw
ssh -i $env:USERPROFILE\.ssh\ssh_sceneagentX_1 -p PORT root@PUBLIC_IP
```

The private key is `%USERPROFILE%\.ssh\ssh_sceneagentX_1`. The Vast API key stays in `%USERPROFILE%\.config\vastai\vast_api_key`. Neither belongs in git.

## Current status

Last lookup that was running: instance `55073623`, public IP `184.144.154.180`, direct port `44997`, proxy `ssh -p 34206 root@ssh1.vast.ai`. GPU is an RTX 4090 with 24564 MiB. The host disk is a fresh rental. The previous disk is not on it.

On that machine, as of the last successful jobs:

- Micromamba 2.9.0 is installed. `bash setup.sh --no-status` finished in about 6.6 minutes.
- `python tools/check_env.py --obj scan` passed: one GPU, driver 580.82.9, all seven envs saw CUDA, `gsplat` and splatfacto imported.
- CUDA toolkit 13.0.88 is at `/usr/local/cuda-13.0` and is the default `nvcc`. The image originally shipped CUDA 12.1.
- Grounded-SAM-2 is cloned under `/root/SceneagentX/third_party/Grounded-SAM-2`. The `gsam2` env has PyTorch `2.14.1+cu130`. The native GroundingDINO `_C` extension loaded after the toolkit upgrade, and native SAM labeled `camera` on all 8 synthetic views.
- A Hugging Face fallback also succeeded earlier: `IDEA-Research/grounding-dino-tiny` and `facebook/sam-vit-base`, prompt `camera.`, 8 views. Those masks are on the laptop in `data/scan/masks/`.
- DINOv2 ViT-S/14 and OpenCLIP ViT-B-32 wrote `/root/SceneagentX/data/scan/codebook.json` (32 dimensions). The laptop copy has `dinov2: true` and `openclip: true`.
- PartField is cloned. Checkpoint `model/model_objaverse.ckpt` downloaded. Inference wrote `exp_results/partfield_features/scan/part_feat_scan_0.npy` with shape `(612280, 448)`. It did not choose a part count. The laptop `parts.json` is still one body.
- Each env activates the CUDA toolkit that matches its PyTorch wheel. `sa` and `gsam2` use CUDA 13.0.88 with torch `2.14.1+cu130`. `nerf`, `partfield`, and `trellis` use CUDA 12.1. `trellis` torch is `2.4.0+cu121` so Kaolin and xformers have a wheel. `transformers` in that env is pinned to `4.46.3`.
- TRELLIS is cloned at `/root/SceneagentX/third_party/TRELLIS`. It wrote three sisters from the eight synthetic views. Those meshes are in `data/scan/sisters/`. Isaac Lab and a video COLMAP run have not been started. There is no `data/scan/raw/orbit1.mp4`.

The laptop changelog tally after those copies is **5 Achieved, 5 Partial, 3 Not yet**.

## Machines used

| Instance | What happened |
| --- | --- |
| `54918067` | First rental. Direct SSH `174.164.26.93:40140` was refused. The public key was not on the instance. |
| `55058403` | Next rental at `192.234.50.153:6525`. Setup and `check_env` passed. SAM install was still downloading when the host stopped scheduling the GPU (`Required resources are currently unavailable`). |
| `55073623` | Current on-demand rental, about $0.636 per hour. Offer id was `53777448`. Image `pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel`, 250 GB disk, label `sceneagent`. Setup was repeated from scratch. |

Search filter that found it: one RTX 4090, 24 GB, at least 24 CPU cores, 64 GB RAM, 250 GB disk, a direct port, driver 580 or newer, reliability above 0.98, CUDA 12.1 or newer. Eighteen on-demand offers came back. Eight matched. The cheapest was rented. Do not rent another unless this one is gone.

## What was installed

1. Micromamba, because `setup.sh` exits without it. The PyTorch image has `/opt/conda/bin/python3` and does not include micromamba.
2. The seven envs in `env/`: `sa`, `nerf`, `gsam2`, `2dgs`, `partfield`, `physgauss`, `trellis`. Each pins `torch==2.1.2+cu121` in the yml. Later pip installs upgraded `gsam2` and `sa` to torch 2.14.1 built for CUDA 13.
3. Grounded-SAM-2 from `https://github.com/IDEA-Research/Grounded-SAM-2`, then `pip install -e .` in `gsam2`.
4. Checkpoints: `checkpoints/sam2.1_hiera_large.pt` (about 856 MB) and `gdino_checkpoints/groundingdino_swint_ogc.pth` (about 661 MB).
5. Extra Python packages in `gsam2`: `opencv-python-headless`, `transformers==4.46.3`, `supervision`, `timm`, `addict`, `yapf`, `pycocotools`. Transformers 5.x was installed first and then downgraded.
6. CUDA toolkit 13.0.88, so `nvcc` matches the CUDA 13 PyTorch wheel.
7. OpenCLIP (`open_clip_torch`) and DINOv2 via `torch.hub` in the `sa` env. That pip install also upgraded `sa` torch to 2.14.1.
8. PartField from `https://github.com/nv-tlabs/PartField`, plus `model_objaverse.ckpt` from `mikaelaangel/partfield-ckpt`. Lightning, h5py, yacs, trimesh, scikit-image, and related packages were installed in the `partfield` env. The full PartField README stack (PyTorch 2.4 and CUDA 12.4) was not installed.

The repo on the VM is a tar snapshot at `/root/SceneagentX`. `.git`, `.venv`, and `data/` were excluded. Synthetic views were copied afterward to `data/scan/renders/synthetic/`. There is no git history on the VM.

## Errors

**SSH.** `174.164.26.93:40140` returned `Permission denied (publickey)`, then later `Connection refused` after the instance stopped. `192.234.50.153:6525` worked, then that instance exited and `vastai start instance` stayed queued with `Required resources are currently unavailable`. The new host `184.144.154.180:44997` first failed with `Host key verification failed` until `StrictHostKeyChecking=accept-new`.

**PowerShell and SSH.** `$()` and `&` inside a double-quoted remote command are eaten by PowerShell. `nohup` then saw no operand, and `ps -p $(cat /root/setup.pid)` looked for `C:\root\setup.pid`. Put remote scripts in files, `sed` away CRLF, and run `bash` on the file.

**CUDA mismatch.** `nvcc` was 12.1. SAM-2's pip install pulled `torch 2.14.1+cu130`. `setup.py build_ext` raised: detected CUDA 12.1 mismatches PyTorch compiled with 13.0. GroundingDINO then warned `Failed to load custom C++ ops` and crashed with `NameError: name '_C' is not defined`. Rebuilds with `g++-10` and `g++-11` also failed before the CUDA 13.0.88 toolkit was installed. After that toolkit, `_C` loaded.

**GroundingDINO Python.** In order: `No module named cv2`, `No module named supervision`, then Transformers 5 removed `BertModel.get_extended_attention_mask`. Pinning `transformers==4.46.3` got past that. The first Hugging Face call used `threshold=` and this version wants `box_threshold=`.

**PartField.** `partfield_inference.py` imported `pytz` and failed. A later retry after `boto3` failed. Mesh libraries were added and it still failed once. A later run exited 0 and wrote `part_feat_scan_0.npy`. `predict` returned `None`. `run_part_clustering.py` for a point cloud saves a KMeans sweep for every k and does not choose one segmentation. No part count was returned. `parts.json` stays one body. The URDF stays at zero joints.

**Environment drift.** `open_clip_torch` in `sa` uninstalled torch 2.1.2+cu121 and installed 2.14.1. `check_env` had already passed on 2.1.2. The yml pin and the live env are no longer the same.

## What this cost us

- Always call `vastai show instances` before SSH. A stopped instance keeps the old IP in notes and refuses the connection.
- Do not rent a second machine while one is running unless the user says the old one is gone.
- `setup.sh` is safe to repeat on a fresh disk. It does not need the scan ply. It does need micromamba.
- Driver 580 is enough. A 590 driver does not fix `nvcc` versus `torch.version.cuda`.
- For this `.ply`, COLMAP and splatfacto are the wrong next step. The missing Harvard piece is the unreleased per-Gaussian physics prior, plus TRELLIS and Isaac.
- A video still needs `data/<obj>/raw/orbit1.mp4` before COLMAP.

## Still not on the VM

- Isaac Lab.
- Isaac Lab.
- A chosen PartField part list.
- Git history of `cursor/add-polycam`.
- The laptop's later edits, unless they were in the tar that was copied for this instance.
