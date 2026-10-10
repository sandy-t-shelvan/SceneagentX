"""COLMAP mask projection keeps the front surface and the training-frame scale."""

from __future__ import annotations

import numpy as np

from tools.colmap_lift import distort_opencv, project_points, select_front, training_c2w


def test_training_pose_scales_translation_only():
    matrix = np.eye(4)
    matrix[:3, 3] = (2.0, 0.0, 0.0)
    posed = training_c2w(matrix, np.eye(3, 4), 0.5)
    assert posed[:3, 3].tolist() == [1.0, 0.0, 0.0]
    assert np.allclose(posed[:3, :3], np.eye(3))


def test_opengl_point_in_front_lands_on_the_principal_point():
    pixels, depth = project_points(
        np.array([[0.0, 0.0, -2.0]]),
        np.eye(4),
        100.0,
        120.0,
        50.0,
        40.0,
        (0.0, 0.0, 0.0, 0.0),
        "opengl",
    )
    assert depth[0] == 2.0
    assert pixels[0, 0] == 50.0
    assert pixels[0, 1] == 40.0


def test_distortion_is_identity_at_the_origin():
    x, y = distort_opencv(np.array([0.0]), np.array([0.0]), 0.2, -0.1, 0.01, -0.01)
    assert x[0] == 0.0
    assert y[0] == 0.0


def test_front_surface_drops_the_far_point():
    valid = np.array([True, True])
    ui = np.array([1, 1])
    vi = np.array([2, 2])
    depth = np.array([1.0, 5.0])
    scales = np.array([0.0, 0.0])
    chosen = select_front(valid, ui, vi, depth, scales, height=4, width=4, relative=0.05)
    assert chosen.tolist() == [True, False]
