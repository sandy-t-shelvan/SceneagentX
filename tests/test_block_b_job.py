"""Shared Block B job. A new object uses the same guards as the can rerun."""

from __future__ import annotations

import importlib.util
import json

import numpy as np
import pytest

from tools.block_b_job import (
    caption_from_spec,
    codebook_collapsed,
    detector_caption,
    grounding_prompt,
    host_launch_script,
    require_object_name,
    union_target_mask,
    write_host_text,
)
from tools.can_instances import capture_for, write_instance_record
from tools.common.paths import repo_root

LAMP_ORBIT = ("orbit1", "/root/SceneagentX/outputs/lamp_orbit1/splatfacto/run/config.yml")
CAN_ORBIT = ("orbit1", "/root/SceneagentX/outputs/can_orbit1/splatfacto/run/config.yml")
SYNONYM_LINE = "aluminum soda can . cylindrical beverage can . drink can with a metal pull tab ."


def test_a_new_object_gets_one_phrase_and_installs_open_clip():
    script = host_launch_script("lamp", [LAMP_ORBIT])
    assert "PROMPT='lamp.'" in script
    assert "pip install open_clip_torch" in script
    assert "micromamba run -n sa" in script or '"$MM" run -n sa pip install open_clip_torch' in script
    b2 = next(line for line in script.splitlines() if "object_langsplat_b2.py" in line)
    assert b2.strip().startswith('"$SA"')
    assert "--obj lamp" in b2
    assert "STOP_AFTER_B2" in script
    assert script.index("STOP_AFTER_B2") < script.index("object_langsplat_b3.py")
    assert "data/lamp/" in script
    assert "{{" not in script


def test_codebook_collapse_and_room_prompt_rules():
    assert codebook_collapsed(0.0, 0.07) is True
    assert codebook_collapsed(0.2, 1.0) is True
    assert codebook_collapsed(0.2, 0.07) is False
    assert grounding_prompt("mug") == "mug."
    assert require_object_name("vaseline") == "vaseline"
    assert grounding_prompt("vaseline") == "vaseline."
    assert require_object_name("bottle") == "bottle"
    assert grounding_prompt("bottle") == "bottle."
    with pytest.raises(ValueError):
        host_launch_script("lamp", [])


def test_room_spec_detects_the_named_target_only():
    spec = {
        "prompt": "a room; I want sisters of Y",
        "job": "room",
        "target": "Y",
        "seen": ["X", "Y", "Z"],
        "deliverable": "sisters_of_target",
    }
    assert caption_from_spec(spec) == "Y."
    script = host_launch_script("kitchen", [LAMP_ORBIT], spec=spec)
    assert "PROMPT='Y.'" in script
    assert "TARGET='Y'" in script
    assert "PROMPT='kitchen" not in script
    line = next(item for item in script.splitlines() if "object_semantics.py" in item)
    assert "--target" in line
    assert "X" not in line
    assert "Z" not in line
    bottle = host_launch_script(
        "bottle",
        [("orbit1", "/root/SceneagentX/outputs/bottle_orbit1/splatfacto/run/config.yml")],
        spec={"prompt": "orbit of a bottle", "job": "orbit", "target": "bottle", "seen": ["bottle"], "deliverable": "sisters_of_target"},
    )
    assert "PROMPT='bottle.'" in bottle
    assert "--obj bottle" in bottle


def test_synonym_line_collapses_to_one_phrase(tmp_path):
    assert detector_caption(SYNONYM_LINE) == "aluminum soda can."
    assert grounding_prompt("mug") == "mug."
    spec = {
        "prompt": "a room; I want sisters of Y",
        "job": "room",
        "target": "Y",
        "seen": ["X", "Y", "Z"],
        "deliverable": "sisters_of_target",
    }
    assert caption_from_spec(spec) == "Y."
    listed = dict(spec)
    listed["target"] = SYNONYM_LINE
    assert caption_from_spec(listed) == "aluminum soda can."
    script = host_launch_script("can", [CAN_ORBIT], prompt=SYNONYM_LINE)
    assert "PROMPT='aluminum soda can.'" in script
    assert "cylindrical" not in script
    from_spec = host_launch_script("can", [CAN_ORBIT], spec=listed)
    assert "PROMPT='aluminum soda can.'" in from_spec
    stored = (repo_root() / "configs" / "can_block_b_prompt.txt").read_text(encoding="utf-8").strip()
    assert stored == "aluminum soda can."

    folder = tmp_path / "data" / "can"
    folder.mkdir(parents=True)
    (folder / "request.json").write_text(
        json.dumps(
            {
                "prompt": SYNONYM_LINE,
                "job": "orbit",
                "target": "can",
                "seen": ["can"],
                "deliverable": "sisters_of_target",
            }
        ),
        encoding="utf-8",
    )
    (folder / "mask_retry.json").write_text(
        json.dumps(
            {
                "detection": "pending",
                "orbits": ["orbit1"],
                "params": {"text_prompt": SYNONYM_LINE},
            }
        ),
        encoding="utf-8",
    )
    ingest = folder / "ingest"
    ingest.mkdir()
    (ingest / "block_a.json").write_text(
        json.dumps({"orbits": [{"orbit": "orbit1", "config": CAN_ORBIT[1]}]}),
        encoding="utf-8",
    )
    from tools.pipeline import mask_retry_commands

    commands = mask_retry_commands("can", tmp_path)
    detect = commands[0]
    assert detect[detect.index("--prompt") + 1] == "aluminum soda can."
    assert "cylindrical" not in " ".join(detect)
    lift = commands[1]
    assert lift[lift.index("--prompt") + 1] == "aluminum soda can."


def test_containing_same_name_box_is_dropped():
    nested = np.zeros((3, 6, 6), dtype=bool)
    nested[0, :, :] = True
    nested[1, 2:4, 2:4] = True
    nested[2, 0, 0] = True
    merged, kept = union_target_mask(nested, ["can", "can", "wall"], "can")
    assert kept == ["can"]
    assert int(merged.sum()) == 4
    assert bool(merged[2, 2]) is True
    assert bool(merged[0, 0]) is False

    separate = np.zeros((2, 4, 4), dtype=bool)
    separate[0, 0, 0] = True
    separate[1, 3, 3] = True
    merged_sep, kept_sep = union_target_mask(separate, ["can", "can"], "can")
    assert kept_sep == ["can", "can"]
    assert bool(merged_sep[0, 0]) is True
    assert bool(merged_sep[3, 3]) is True
    assert int(merged_sep.sum()) == 2

    alone = np.ones((1, 4, 4), dtype=bool)
    merged_one, kept_one = union_target_mask(alone, ["can"], "can")
    assert kept_one == ["can"]
    assert bool(merged_one.all()) is True

    close = np.zeros((2, 10, 10), dtype=bool)
    close[0, :, :] = True
    close[1, :9, :] = True
    merged_close, kept_close = union_target_mask(close, ["can", "can"], "can")
    assert kept_close == ["can", "can"]
    assert int(merged_close.sum()) == 100

    tighter = np.zeros((2, 10, 10), dtype=bool)
    tighter[0, :, :] = True
    tighter[1, :8, :] = True
    merged_tight, kept_tight = union_target_mask(tighter, ["can", "can"], "can")
    assert kept_tight == ["can"]
    assert int(merged_tight.sum()) == 80


def test_host_text_survives_a_windows_console():
    class Stream:
        encoding = "cp1252"

        def __init__(self) -> None:
            self.parts: list[str] = []

        def write(self, text: str) -> None:
            text.encode("cp1252")
            self.parts.append(text)

    stream = Stream()
    write_host_text(stream, "loaded \u2705\n")
    assert stream.parts
    assert "loaded" in stream.parts[0]


def test_capture_record_uses_the_object_folder(tmp_path):
    folder = tmp_path / "lamp"
    (folder / "raw").mkdir(parents=True)
    (folder / "raw" / "orbit1.MOV").write_bytes(b"x")
    record = capture_for(folder)
    assert record["used_for_eval"] == "data/lamp/raw/orbit1.MOV"
    assert record["orbits"][0]["exists"] is True
    assert record["orbits"][1]["exists"] is False
    written = write_instance_record(
        folder,
        orbit="orbit2",
        clustered={"instances": [], "meta": {"n_object": 0}},
        mask_lift={"n_gaussians": 1, "n_inside": 0, "n_outside": 1},
    )
    assert written["capture"]["used_for_eval"] == "data/lamp/raw/orbit1.MOV"
    assert written["codebook"] == "data/lamp/codebook_orbit2.json"
    assert "can" not in written["instance_count_note"]


def test_can_launcher_uses_the_shared_job():
    path = repo_root() / "vast-cluster" / "scripts" / "launch_can_block_b_rerun.py"
    spec = importlib.util.spec_from_file_location("launch_can_block_b_rerun", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = module.START
    assert "pip install open_clip_torch" in text
    assert "STOP_AFTER_B2" in text
    assert '"$SA" -u vast-cluster/scripts/object_langsplat_b2.py --obj can' in text
    assert text.index("STOP_AFTER_B2") < text.index("object_langsplat_b3.py")
    assert "aluminum soda can" in text


def test_autoencoder_last_layer_is_linear():
    pytest.importorskip("torch")
    pytest.importorskip("PIL")
    path = repo_root() / "vast-cluster" / "scripts" / "object_langsplat_b2.py"
    spec = importlib.util.spec_from_file_location("object_langsplat_b2", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.LangSplatAutoencoder()
    import torch

    assert isinstance(list(model.encoder.children())[-1], torch.nn.Linear)
    assert isinstance(list(model.decoder.children())[-1], torch.nn.Linear)
