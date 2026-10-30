# Polycam → 3D Gaussian Splatting Workflow

This guide shows how to use Polycam scans as an alternative to video capture for the SceneAgent pipeline, bypassing B1-B2 failures due to poor video quality.

## Overview

**Standard SceneAgent:**
```
Video Capture → B1 (Frame Check) → B2 (COLMAP) → B3 (Splatfacto) → B4+ (Continue)
                   ❌ Failed            ❌ Failed
```

**Polycam Alternative:**
```
Polycam Scan → Import → B3 (Splatfacto) → B4 (Manual Scale) → B5+ (Continue)
               ✅ Works    ✅ Works         ✅ Works
```

## Prerequisites

### 1. Install Dependencies

```bash
pip install -r requirements-sa.txt  # Now includes trimesh
```

### 2. Polycam App Setup

- Install Polycam app on your iPhone/iPad (supports LiDAR devices)
- Or use Polycam on Android with photogrammetry mode
- Free tier works, but Pro gives better exports

## Step-by-Step Guide

### Step 1: Capture with Polycam

#### Option A: LiDAR Scan (Recommended - iPhone 12 Pro+, iPad Pro)

1. Open Polycam app
2. Select **"LiDAR"** mode
3. Scan your marker object:
   - Walk around object in circular path
   - Keep device 1-2 feet away
   - Capture 360° coverage
   - Takes ~2-3 minutes
4. Wait for processing to complete

#### Option B: Photo Mode (Any phone with camera)

1. Open Polycam app
2. Select **"Photo"** mode
3. Take 50-100 photos around object:
   - Circular orbit around object
   - Overlap between photos
   - Consistent lighting
   - Takes ~5-10 minutes total

### Step 2: Export from Polycam

1. Open your scan in Polycam
2. Tap **"Export"** button
3. Choose export format:

**Best Option: "COLMAP" format**
```
✅ Direct COLMAP export
   - Includes sparse/0/ directory
   - Camera poses already computed
   - Ready to import
```

**Alternative: "All Data" or "Raw Data"**
```
✅ Includes cameras.json
✅ Includes mesh (.ply or .obj)
✅ Includes images/
   - Requires format conversion
   - Import script handles this
```

4. Export to Files app or email to yourself
5. Transfer to your computer

### Step 3: Organize Files

Place exported Polycam data in your project:

```bash
# Create directory
mkdir -p data/marker/polycam

# Move Polycam export there
# Structure should look like:
data/marker/polycam/
├── sparse/              # If COLMAP export
│   └── 0/
│       ├── cameras.txt
│       ├── images.txt
│       └── points3D.txt
├── images/              # All images
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── cameras.json         # If LiDAR/Photo export
└── mesh.ply             # Or mesh.obj
```

### Step 4: Measure Your Object

**Important:** Since we don't have AprilTag, measure manually with ruler:

```bash
# Measure marker body length in meters
# Example: marker is 14 cm = 0.14 m
python -m tools.common.status set --obj marker ground_truth.length_m 0.14
```

### Step 5: Import Polycam Data

```bash
# Import Polycam scan for orbit 1
python tools/import_polycam.py \
  --obj marker \
  --scan data/marker/polycam/ \
  --orbit 1 \
  --scale 0.14
```

**What this does:**
- ✅ Detects Polycam export format automatically
- ✅ Converts to COLMAP format if needed
- ✅ Copies images to `data/marker/frames/orbit1/`
- ✅ Creates sparse reconstruction in `data/marker/colmap/orbit1/`
- ✅ Saves manual scale to status.json

**Expected output:**
```
✓ Imported 87 images from Polycam scan
  Format: polycam_lidar
  Source: data/marker/polycam
  Manual scale: 0.14 m

Next steps:
  python tools/step02_splatfacto.py --obj marker --orbit 1
```

### Step 6: Train 3D Gaussians (Splatfacto)

```bash
# On GPU machine with nerfstudio installed
python tools/step02_splatfacto.py --obj marker --orbit 1
```

**What this does:**
- Trains splatfacto on Polycam camera poses
- Creates 3D Gaussian Splatting representation
- Outputs to `data/marker/colmap/orbit1/splat.ply`
- Takes 15-30 minutes on RTX 4090

**Expected gates (B3):**
- PSNR ≥ 28 ✓
- SSIM ≥ 0.90 ✓
- 300k-3M Gaussians ✓

### Step 7: Alignment with Manual Scale

```bash
# No multi-orbit alignment needed for Polycam workflow
python tools/align.py \
  --obj marker \
  --manual-scale 0.14 \
  --no-rerun
```

**What this does:**
- ✅ Uses manual scale instead of AprilTag
- ✅ Skips multi-orbit alignment (only orbit 1)
- ✅ Marks B4 as passed

### Step 8: Continue Pipeline

```bash
# Continue with semantic processing (B5+)
python -m tools.pipeline --obj marker --from B5
```

Now the rest of the SceneAgent pipeline continues normally:
- B5: Grounded-SAM-2 segmentation
- B6: Lift masks to Gaussians
- B7: Feature codebooks
- B8+: Mesh, parts, physics, etc.

## Comparison: Video vs Polycam

| Aspect | Video Capture | Polycam Scan |
|--------|--------------|--------------|
| **Setup time** | 30-60 min (lighting, tripod) | 2-3 min (phone scan) |
| **Capture time** | 3 orbits × 1 min each | Single 3-min scan |
| **Equipment** | Camera + tripod + lights | Just your phone |
| **Lighting needs** | Critical (even, no glare) | Flexible |
| **AprilTag** | Required (print + place) | Not needed |
| **Failure rate** | High (blur, exposure, tag) | Low |
| **Reconstruction quality** | Excellent (if works) | Very good |
| **Number of frames** | 150-250 per orbit | 50-100 total |
| **Multi-orbit** | Yes (orbit 1,2,3) | Single scan |
| **Scale accuracy** | Automatic (AprilTag) | Manual (±1-2mm) |
| **Code changes** | None (standard) | Import script added |

## Troubleshooting

### Import fails: "Could not detect Polycam format"

**Problem:** Export doesn't have expected files.

**Solution:**
```bash
# Check what you have
ls -la data/marker/polycam/

# Should have ONE of:
# - sparse/0/ directory (COLMAP export)
# - cameras.json + mesh.ply/obj (LiDAR/Photo)
# - images/ folder with photos
```

### Import fails: "No mesh found"

**Problem:** Mesh file missing or wrong format.

**Solution:**
- Re-export from Polycam with "All Data" option
- Ensure `.ply` or `.obj` file is included
- Check file isn't corrupted (should be >1 MB)

### Splatfacto training fails

**Problem:** COLMAP format issues or bad camera poses.

**Solution:**
```bash
# Verify COLMAP files exist
ls data/marker/colmap/orbit1/sparse/cameras.txt
ls data/marker/colmap/orbit1/sparse/images.txt

# Check frame count
ls data/marker/frames/orbit1/ | wc -l
# Should have 50+ images

# Try with nerfstudio's validator
ns-process-data --data data/marker/colmap/orbit1/ --verbose
```

### Low PSNR/SSIM scores

**Problem:** Polycam scan quality not sufficient.

**Solutions:**
1. **Re-scan with better coverage:**
   - More complete 360° orbit
   - Slower movement
   - Better lighting

2. **Use Photo mode instead of LiDAR:**
   - Take 80-100 photos
   - More overlap between photos
   - Process takes longer but better quality

3. **Clean up scan in Polycam:**
   - Use Polycam's "Edit" tools
   - Remove background/artifacts
   - Re-export after cleaning

### Scale inaccuracy

**Problem:** Manual measurement has error.

**Solution:**
```bash
# Measure more carefully with caliper/ruler
# Then update:
python -m tools.common.status set --obj marker ground_truth.length_m 0.142

# Re-run alignment
python tools/align.py --obj marker --manual-scale 0.142 --no-rerun
```

## Advanced: Hybrid Workflow

If you have both video and Polycam:

```bash
# Use Polycam poses but video frames for texture
python tools/import_polycam.py --obj marker --scan data/marker/polycam/ --orbit 1

# Copy video frames to same directory
python tools/step00_check_frames.py --obj marker --orbit 1

# Manually merge frames folders
# Then train splatfacto with more data
python tools/step02_splatfacto.py --obj marker --orbit 1
```

## File Outputs

After Polycam workflow completes:

```
data/marker/
├── polycam/                    # Original Polycam export
│   └── ...
├── frames/orbit1/              # Imported images
│   ├── frame_0000.jpg
│   └── ...
├── colmap/orbit1/              # COLMAP format
│   ├── sparse/
│   │   ├── cameras.txt
│   │   ├── images.txt
│   │   └── points3D.txt
│   └── splat.ply              # 3D Gaussians (after B3)
├── renders/
│   └── step02/                # Splatfacto renders
└── status.json                # Updated with Polycam metadata
```

## Next Steps

Once B4 passes with Polycam data:
- All downstream blocks (B5-B17) work normally
- Segmentation, physics, articulation proceed as designed
- No further modifications needed

## Questions?

- Check `tools/import_polycam.py` source for format details
- See `tools/align.py` for manual scale implementation
- Original SceneAgent paper: [link]
- Polycam docs: https://poly.cam/docs

---

**Summary:** This workflow lets you bypass video capture issues by using Polycam's reliable 3D scanning, then continues with standard SceneAgent pipeline for semantic understanding and physics simulation.
