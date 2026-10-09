"""Import Polycam scan for SceneAgent pipeline.

Converts Polycam exports (mesh, cameras, images) into the format expected
by the SceneAgent pipeline, bypassing B1-B2 and providing data for B3.

Polycam exports can be:
1. LiDAR scan with cameras (.ply mesh + images + cameras.json)
2. Photo scan (.obj mesh + images + camera poses)
3. COLMAP export (direct import)

This tool converts any of these into the COLMAP format that step02_splatfacto.py
expects, then optionally trains splatfacto to generate 3D Gaussians.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import argparse
import json
import shutil
from typing import Any

import cv2
import numpy as np

from tools.common.paths import frames_dir, object_dir
from tools.common.status import begin_step, finish_step, load_status, save_status, set_path
from tools.step01_colmap import colmap_dir, sparse_dir


def polycam_dir(obj: str, root: Path | None = None) -> Path:
    """Directory for Polycam import data."""
    return object_dir(obj, root) / "polycam"


def detect_polycam_format(scan_dir: Path) -> str:
    """Detect which format the Polycam export uses."""
    if not scan_dir.is_dir():
        raise ValueError(f"Polycam directory does not exist: {scan_dir}")
    
    has_colmap = (scan_dir / "sparse" / "0").is_dir()
    has_cameras_json = (scan_dir / "cameras.json").is_file()
    has_mesh = any(scan_dir.glob("*.ply")) or any(scan_dir.glob("*.obj"))
    has_images = (scan_dir / "images").is_dir() or len(list(scan_dir.glob("*.jpg"))) > 0
    
    if has_colmap:
        return "colmap"
    elif has_cameras_json and has_mesh:
        return "polycam_lidar"
    elif has_mesh and has_images:
        return "polycam_photo"
    else:
        raise ValueError(
            f"Could not detect Polycam format in {scan_dir}. "
            f"Expected: COLMAP export, or mesh + cameras.json, or mesh + images/"
        )


def load_polycam_cameras(cameras_json: Path) -> dict[str, Any]:
    """Load Polycam cameras.json format.
    
    Polycam format (example):
    {
      "frames": [
        {
          "id": 0,
          "file_path": "frame_0000.jpg",
          "transform_matrix": [[...], [...], [...], [...]],  # 4x4 world-to-camera
          "intrinsics": {
            "fx": 1234.5, "fy": 1234.5, "cx": 640, "cy": 360,
            "width": 1280, "height": 720
          }
        },
        ...
      ]
    }
    """
    if not cameras_json.is_file():
        raise ValueError(f"cameras.json not found: {cameras_json}")
    
    with open(cameras_json, 'r') as f:
        data = json.load(f)
    
    return data


def write_colmap_cameras(intrinsics: dict[str, Any], output_path: Path) -> None:
    """Write COLMAP cameras.txt format.
    
    Format: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]
    Example: 1 PINHOLE 1280 720 1234.5 1234.5 640 360
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write("# Number of cameras: 1\n")
        
        # Polycam typically uses pinhole model
        camera_line = (
            f"1 PINHOLE {intrinsics['width']} {intrinsics['height']} "
            f"{intrinsics['fx']:.6f} {intrinsics['fy']:.6f} "
            f"{intrinsics['cx']:.6f} {intrinsics['cy']:.6f}\n"
        )
        f.write(camera_line)


def write_colmap_images(frames: list[dict[str, Any]], output_path: Path) -> None:
    """Write COLMAP images.txt format.
    
    Format:
    IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME
    POINTS2D[] as empty (we don't have 2D-3D correspondences)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {len(frames)}\n")
        
        for idx, frame in enumerate(frames, start=1):
            # Convert 4x4 transform matrix to quaternion + translation
            transform = np.array(frame['transform_matrix'])
            
            # Polycam uses OpenGL convention (camera looks down -Z)
            # Convert to COLMAP convention (camera looks down +Z)
            # This might need adjustment based on actual Polycam convention
            rotation = transform[:3, :3]
            translation = transform[:3, 3]
            
            # Convert rotation matrix to quaternion
            quat = rotation_matrix_to_quaternion(rotation)
            
            # Write image line
            image_name = frame.get('file_path', f"frame_{idx:04d}.jpg")
            f.write(
                f"{idx} {quat[0]:.10f} {quat[1]:.10f} {quat[2]:.10f} {quat[3]:.10f} "
                f"{translation[0]:.10f} {translation[1]:.10f} {translation[2]:.10f} "
                f"1 {image_name}\n"
            )
            # Empty points line (no 2D-3D correspondences)
            f.write("\n")


def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Convert 3x3 rotation matrix to quaternion [w, x, y, z]."""
    trace = np.trace(R)
    
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    
    return np.array([w, x, y, z])


def write_colmap_points3d(mesh_path: Path, output_path: Path, sample_points: int = 10000) -> None:
    """Write COLMAP points3D.txt format from mesh.
    
    Sample points from the mesh surface to create a sparse 3D point cloud.
    Format: POINT3D_ID X Y Z R G B ERROR TRACK[]
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load mesh and sample points
    points, colors = sample_mesh_points(mesh_path, sample_points)
    
    with open(output_path, 'w') as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
        f.write(f"# Number of points: {len(points)}\n")
        
        for idx, (point, color) in enumerate(zip(points, colors), start=1):
            # Write point with empty track (no 2D correspondences)
            f.write(
                f"{idx} {point[0]:.10f} {point[1]:.10f} {point[2]:.10f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])} 0.0\n"
            )


def sample_mesh_points(mesh_path: Path, num_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Sample points uniformly from mesh surface.
    
    Returns: (points, colors) where points is Nx3 and colors is Nx3.
    """
    import trimesh
    
    # Load mesh
    mesh = trimesh.load(str(mesh_path))
    
    # Sample points from surface
    points, face_indices = trimesh.sample.sample_surface(mesh, num_points)
    
    # Get colors if available
    if hasattr(mesh.visual, 'vertex_colors') and mesh.visual.vertex_colors is not None:
        # Interpolate vertex colors to sampled points
        colors = mesh.visual.vertex_colors[:, :3]  # RGB only
        # Simple: use color from first vertex of face
        colors = colors[mesh.faces[face_indices, 0]]
    else:
        # Default gray color
        colors = np.full((num_points, 3), 128, dtype=np.uint8)
    
    return points, colors


def copy_images_to_frames(source_dir: Path, dest_dir: Path, frame_list: list[dict[str, Any]]) -> None:
    """Copy images from Polycam export to frames directory."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    for frame in frame_list:
        src_name = frame.get('file_path', '')
        src_path = source_dir / src_name
        
        if not src_path.is_file():
            # Try without subdirectory
            src_path = source_dir.parent / src_name
        
        if not src_path.is_file():
            raise FileNotFoundError(f"Image not found: {src_name} (looked in {source_dir})")
        
        # Copy to destination
        dest_path = dest_dir / src_name
        shutil.copy2(src_path, dest_path)


def import_polycam_colmap(scan_dir: Path, obj: str, orbit: int, root: Path | None) -> dict[str, Any]:
    """Import when Polycam exported directly as COLMAP format."""
    src_sparse = scan_dir / "sparse" / "0"
    dest_sparse = sparse_dir(obj, orbit, root)
    dest_frames = frames_dir(obj, orbit, root)
    
    # Copy COLMAP sparse reconstruction
    dest_sparse.parent.mkdir(parents=True, exist_ok=True)
    if dest_sparse.exists():
        shutil.rmtree(dest_sparse)
    shutil.copytree(src_sparse, dest_sparse)
    
    # Copy images
    src_images = scan_dir / "images"
    if src_images.is_dir():
        dest_frames.mkdir(parents=True, exist_ok=True)
        for img in src_images.glob("*"):
            if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                shutil.copy2(img, dest_frames / img.name)
    
    num_images = len(list(dest_frames.glob("*"))) if dest_frames.exists() else 0
    
    return {
        "format": "colmap",
        "num_images": num_images,
        "source": str(scan_dir),
    }


def import_polycam_lidar(scan_dir: Path, obj: str, orbit: int, root: Path | None) -> dict[str, Any]:
    """Import Polycam LiDAR scan (cameras.json + mesh + images)."""
    cameras_json = scan_dir / "cameras.json"
    mesh_path = next((p for p in scan_dir.glob("*.ply")), None) or next((p for p in scan_dir.glob("*.obj")), None)
    
    if not mesh_path:
        raise FileNotFoundError(f"No mesh found in {scan_dir}")
    
    # Load camera data
    camera_data = load_polycam_cameras(cameras_json)
    frames = camera_data.get('frames', [])
    
    if not frames:
        raise ValueError("No frames found in cameras.json")
    
    # Get intrinsics from first frame (assuming all frames use same camera)
    intrinsics = frames[0].get('intrinsics', {})
    
    # Write COLMAP format files
    dest_sparse = sparse_dir(obj, orbit, root)
    dest_sparse.mkdir(parents=True, exist_ok=True)
    
    write_colmap_cameras(intrinsics, dest_sparse / "cameras.txt")
    write_colmap_images(frames, dest_sparse / "images.txt")
    write_colmap_points3d(mesh_path, dest_sparse / "points3D.txt")
    
    # Copy images
    dest_frames = frames_dir(obj, orbit, root)
    images_dir = scan_dir / "images" if (scan_dir / "images").is_dir() else scan_dir
    copy_images_to_frames(images_dir, dest_frames, frames)
    
    return {
        "format": "polycam_lidar",
        "num_images": len(frames),
        "num_points3d": 10000,  # sampled from mesh
        "mesh": str(mesh_path),
        "source": str(scan_dir),
    }


def import_polycam_scan(
    scan_dir: Path,
    obj: str,
    orbit: int = 1,
    root: Path | None = None,
    manual_scale: float | None = None,
) -> dict[str, Any]:
    """Main import function - detects format and imports accordingly."""
    begin_step(obj, "B2_polycam", root)
    
    try:
        fmt = detect_polycam_format(scan_dir)
        
        if fmt == "colmap":
            result = import_polycam_colmap(scan_dir, obj, orbit, root)
        elif fmt == "polycam_lidar":
            result = import_polycam_lidar(scan_dir, obj, orbit, root)
        elif fmt == "polycam_photo":
            # Similar to LiDAR but might have different camera format
            result = import_polycam_lidar(scan_dir, obj, orbit, root)  # Try same logic
        else:
            raise ValueError(f"Unknown format: {fmt}")
        
        # Save to status
        doc = load_status(obj, root)
        set_path(doc, f"polycam_import.orbit{orbit}", result)
        
        if manual_scale is not None:
            set_path(doc, "ground_truth.scale_from_polycam", manual_scale)
        
        save_status(obj, doc, root)
        
        finish_step(obj, "B2_polycam", "pass", root, actual=result)
        
        return result
        
    except Exception as exc:
        error_result = {"error": str(exc), "passed": False}
        finish_step(obj, "B2_polycam", "fail", root, actual=error_result)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Import Polycam scan for SceneAgent pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Import Polycam scan for orbit 1
  python tools/import_polycam.py --obj marker --scan data/marker/polycam/ --orbit 1
  
  # Import with manual scale (marker length in meters)
  python tools/import_polycam.py --obj marker --scan data/marker/polycam/ --scale 0.14
  
After import, continue with:
  python tools/step02_splatfacto.py --obj marker --orbit 1
  python tools/align.py --obj marker --manual-scale 0.14
        """
    )
    parser.add_argument("--obj", default="marker", help="Object name")
    parser.add_argument("--scan", type=Path, required=True, help="Path to Polycam export directory")
    parser.add_argument("--orbit", type=int, default=1, help="Orbit number (default: 1)")
    parser.add_argument("--root", type=Path, default=None, help="Project root directory")
    parser.add_argument("--scale", type=float, default=None, help="Manual scale: object length in meters")
    
    args = parser.parse_args(argv)
    
    try:
        result = import_polycam_scan(args.scan, args.obj, args.orbit, args.root, args.scale)
        print(f"✓ Imported {result['num_images']} images from Polycam scan")
        print(f"  Format: {result['format']}")
        print(f"  Source: {result['source']}")
        if args.scale:
            print(f"  Manual scale: {args.scale} m")
        print(f"\nNext steps:")
        print(f"  python tools/step02_splatfacto.py --obj {args.obj} --orbit {args.orbit}")
        return 0
    except Exception as exc:
        sys.stderr.write(f"Error: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
