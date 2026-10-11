#!/usr/bin/env python
"""One Isaac render of the bottle mesh from the frame_00113 COLMAP pose.

The picture is written only after a non-empty RGB buffer comes back.
"""

from __future__ import annotations

import json
from pathlib import Path

from isaaclab.app import AppLauncher
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--usda", required=True)
parser.add_argument("--png", required=True)
parser.add_argument("--pose", required=True)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import numpy as np
from PIL import Image
from pxr import Gf, Usd, UsdGeom, UsdLux

import omni.replicator.core as rep


def main() -> None:
    pose = json.loads(Path(args.pose).read_text(encoding="utf-8"))
    stage = Usd.Stage.Open(args.usda)
    if stage is None:
        opened = Usd.Stage.CreateInMemory()
        opened.GetRootLayer().Import(args.usda)
        stage = opened
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    light = UsdLux.DomeLight.Define(stage, "/World/Light")
    light.CreateIntensityAttr(2500.0)
    camera = UsdGeom.Camera.Define(stage, "/World/Camera")
    width = int(pose["width"])
    height = int(pose["height"])
    aperture = 20.955
    camera.CreateHorizontalApertureAttr(aperture)
    camera.CreateVerticalApertureAttr(aperture * height / width)
    camera.CreateFocalLengthAttr(float(pose["fx"]) / width * aperture)
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.01, 100.0))
    matrix = np.asarray(pose["c2w_opengl"], dtype=np.float64)
    transform = Gf.Matrix4d(
        float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2]), float(matrix[0, 3]),
        float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2]), float(matrix[1, 3]),
        float(matrix[2, 0]), float(matrix[2, 1]), float(matrix[2, 2]), float(matrix[2, 3]),
        float(matrix[3, 0]), float(matrix[3, 1]), float(matrix[3, 2]), float(matrix[3, 3]),
    )
    UsdGeom.Xformable(camera).AddTransformOp().Set(transform)
    render_product = rep.create.render_product("/World/Camera", (width, height))
    annotator = rep.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    image = None
    for _ in range(8):
        rep.orchestrator.step()
        data = np.asarray(annotator.get_data())
        if data.size == 0:
            continue
        rgb = data[:, :, :3]
        if int(rgb.std()) < 2:
            continue
        image = rgb
    if image is None:
        raise RuntimeError("Isaac returned no varying RGB frame")
    destination = Path(args.png)
    destination.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image).save(destination)
    print("ISAAC_FRAME_OK", destination, destination.stat().st_size)


try:
    main()
finally:
    simulation_app.close()
