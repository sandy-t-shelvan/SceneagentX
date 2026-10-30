"""Block H takes a video and a prompt. A checkpoint stays unless --fresh is set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.agent_loop import CAN_CAPTION, _skill_ingest, _skill_run_step, main, run_block_h
from tools.block_b_job import detector_caption
from tools.gate_bypass import ENV_NAME
from tools.vast_host import VastHost

PROMPT = "This is an orbit video of the can. I want the can."
PHRASE = "This is an orbit video of the can."


def _parse(prompt: str, images: list[Path]) -> dict[str, object]:
    del images
    return {"job": "orbit", "target": "can", "seen": ["can"], "deliverable": "sisters_of_target"}


def _plant(tmp_path: Path) -> Path:
    for orbit in (1, 2, 3):
        folder = tmp_path / "data" / "can" / "colmap" / f"orbit{orbit}"
        folder.mkdir(parents=True)
        (folder / "splat.ply").write_bytes(b"ply")
        sparse = folder / "sparse" / "0"
        sparse.mkdir(parents=True)
        (sparse / "cameras.bin").write_bytes(b"c")
    config = tmp_path / "outputs" / "can_orbit1" / "splatfacto" / "2020-01-01_000000" / "config.yml"
    config.parent.mkdir(parents=True)
    config.write_text("checkpoint: true\n", encoding="utf-8")
    return config


def test_prompt_is_stored_before_any_skill(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    monkeypatch.setattr("tools.task_spec.parse_pipeline_prompt", _parse)
    order: list[str] = []

    def skill() -> int:
        order.append("skill")
        spec = json.loads((tmp_path / "data" / "can" / "request.json").read_text(encoding="utf-8"))
        assert spec["prompt"] == PROMPT
        assert spec["target"] == "can"
        return 0

    video = tmp_path / "orbit1.MOV"
    video.write_bytes(b"video")
    code = run_block_h(
        "can",
        tmp_path,
        skills=[("SCENEAGENT-RUN-PIPELINE", skill)],
        prompt=PROMPT,
        source=video,
        videos=[video],
    )
    assert code == 0
    assert order == ["skill"]
    assert detector_caption(PROMPT) == PHRASE
    text = capsys.readouterr().out
    assert text.index(f"detector caption {PHRASE}") < text.index("SCENEAGENT-RUN-PIPELINE exit 0")
    status = json.loads((tmp_path / "data" / "can" / "block_h.json").read_text(encoding="utf-8"))
    assert status["prompt"] == PROMPT
    assert status["caption"] == PHRASE
    assert status["videos"] == [str(video)]
    assert status["achieved"] is False
    assert status["m1_met"] is False
    assert status["m2_met"] is False
    assert status["milestone"] is False
    assert "aluminum soda can" not in text


def test_prompt_failure_is_unmet_and_bypass_does_not_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv(ENV_NAME, raising=False)

    def boom() -> int:
        raise AssertionError("a skill ran before the prompt was accepted")

    skills = [("SCENEAGENT-RUN-PIPELINE", boom)]
    code = run_block_h("can", tmp_path, skills=skills, prompt=PROMPT)
    assert code == 2
    spec = json.loads((tmp_path / "data" / "can" / "request.json").read_text(encoding="utf-8"))
    assert spec == {"prompt": PROMPT}
    status = json.loads((tmp_path / "data" / "can" / "block_h.json").read_text(encoding="utf-8"))
    assert status["stopped"] == "pipeline_prompt"
    assert status["caption"] == PHRASE
    assert status["achieved"] is False
    assert status["m1_met"] is False
    assert status["m2_met"] is False
    assert status["milestone"] is False
    assert not (tmp_path / "data" / "can" / "gate_bypass.json").is_file()

    monkeypatch.setenv(ENV_NAME, "1")
    ran: list[str] = []

    def later() -> int:
        ran.append("later")
        return 0

    code = run_block_h("can", tmp_path, skills=[("SCENEAGENT-RUN-PIPELINE", later)], prompt=PROMPT)
    assert code == 0
    assert ran == ["later"]
    status = json.loads((tmp_path / "data" / "can" / "block_h.json").read_text(encoding="utf-8"))
    assert status["achieved"] is False
    assert status["m1_met"] is False
    assert status["m2_met"] is False
    assert status["milestone"] is False
    recorded = json.loads((tmp_path / "data" / "can" / "gate_bypass.json").read_text(encoding="utf-8"))
    assert recorded["achieved"] is False
    assert recorded["passed"] is False
    assert recorded["milestone"] is False
    gate = recorded["gates"][0]
    assert gate["gate"] == "pipeline_prompt"
    assert gate["check"] == "unmet"
    assert gate["passed"] is False


def test_fresh_drops_old_masks_and_keeps_video_and_checkpoints(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("tools.task_spec.parse_pipeline_prompt", _parse)
    folder = tmp_path / "data" / "can"
    masks = folder / "masks" / "orbit1"
    masks.mkdir(parents=True)
    (masks / "object.json").write_text("{}", encoding="utf-8")
    (folder / "mask_retry.json").write_text("{}", encoding="utf-8")
    (folder / "ingest").mkdir()
    (folder / "ingest" / "block_a.json").write_text("{}", encoding="utf-8")
    raw = folder / "raw"
    raw.mkdir(parents=True)
    video = raw / "orbit1.MOV"
    video.write_bytes(b"video")
    config = tmp_path / "outputs" / "can_orbit1" / "splatfacto" / "t" / "config.yml"
    config.parent.mkdir(parents=True)
    config.write_text("checkpoint: true\n", encoding="utf-8")
    ply = folder / "colmap" / "orbit1" / "splat.ply"
    ply.parent.mkdir(parents=True)
    ply.write_bytes(b"ply")
    seen: dict[str, bool] = {}

    def skill() -> int:
        seen["masks"] = (folder / "masks").exists()
        seen["retry"] = (folder / "mask_retry.json").exists()
        seen["ingest"] = (folder / "ingest").exists()
        seen["video"] = video.is_file()
        seen["ply"] = ply.is_file()
        seen["config"] = config.is_file()
        seen["request"] = (folder / "request.json").is_file()
        return 0

    code = run_block_h(
        "can",
        tmp_path,
        skills=[("SCENEAGENT-RUN-PIPELINE", skill)],
        prompt=PROMPT,
        fresh=True,
        videos=[video],
    )
    assert code == 0
    assert seen == {
        "masks": False,
        "retry": False,
        "ingest": False,
        "video": True,
        "ply": True,
        "config": True,
        "request": True,
    }


def test_without_prompt_keeps_the_synonym_caption(tmp_path: Path):
    def skill() -> int:
        assert not (tmp_path / "data" / "can" / "request.json").is_file()
        return 0

    code = run_block_h("can", tmp_path, skills=[("SCENEAGENT-RUN-PIPELINE", skill)])
    assert code == 0
    status = json.loads((tmp_path / "data" / "can" / "block_h.json").read_text(encoding="utf-8"))
    assert status["caption"] == CAN_CAPTION
    assert "prompt" not in status
    assert status["achieved"] is False
    assert status["milestone"] is False


def test_main_passes_prompt_videos_and_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    seen: dict[str, object] = {}

    def fake_run(*_args, **kwargs):
        seen.update(kwargs)
        return 0

    monkeypatch.setattr("tools.agent_loop.run_block_h", fake_run)
    videos = []
    for name in ("orbit1.MOV", "orbit2.MOV", "orbit3.MOV"):
        path = tmp_path / name
        path.write_bytes(b"video")
        videos.append(path)
    argv = ["can", "--orchestrate", "--fresh", "--prompt", PROMPT, "--root", str(tmp_path)]
    for path in videos:
        argv.extend(["--input", str(path)])
    assert main(argv) == 0
    assert seen["prompt"] == PROMPT
    assert seen["fresh"] is True
    assert seen["source"] == videos[0]
    assert seen["videos"] == videos

    seen.clear()
    assert main(["can", "--orchestrate", "--root", str(tmp_path)]) == 0
    assert seen["prompt"] is None
    assert seen["fresh"] is False
    assert seen["videos"] == []


def test_prompt_without_orchestrate_does_not_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    code = main(["can", "--once", "--prompt", PROMPT, "--root", str(tmp_path)])
    assert code == 2
    assert not (tmp_path / "data" / "can" / "request.json").is_file()


def test_checkpoint_stays_unless_fresh(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]):
    config = _plant(tmp_path)
    ply = tmp_path / "data" / "can" / "colmap" / "orbit1" / "splat.ply"
    assert _skill_ingest("can", tmp_path) == 0
    assert "ingest kept existing COLMAP and splat checkpoints" in capsys.readouterr().out
    assert ply.is_file()
    assert config.is_file()

    host = VastHost(instance_id=1, status="running", host="203.0.113.10", port=2222)
    monkeypatch.setattr("tools.pipeline.running_vast_host", lambda: host)
    order: list[str] = []
    scripts: list[str] = []

    def sync(_host) -> int:
        order.append("sync")
        return 0

    def ssh(_host, script: str, timeout: int = 120):
        del timeout
        order.append("ssh")
        scripts.append(script)
        return 0, "", ""

    def push(_host, _obj, root=None) -> int:
        del root
        order.append("push")
        return 0

    def colmap(_obj, orbit, _root, _runner, mark_block=True) -> int:
        del mark_block
        order.append(f"colmap{orbit}")
        return 0

    def splat(_obj, orbit, _root, _runner, mark_block=True) -> int:
        del mark_block
        order.append(f"splat{orbit}")
        return 0

    def pull(_host, _obj, _root, orbit: int) -> int:
        order.append(f"pull{orbit}")
        return 0

    monkeypatch.setattr("tools.agent_loop._sync_host_code", sync)
    monkeypatch.setattr("tools.agent_loop._ssh", ssh)
    monkeypatch.setattr("tools.pipeline.push_missing_raw_videos", push)
    monkeypatch.setattr("tools.pipeline.pull_reconstruction", pull)
    monkeypatch.setattr("tools.step01_colmap.run_orbit", colmap)
    monkeypatch.setattr("tools.step02_splatfacto.run_orbit", splat)
    assert _skill_ingest("can", tmp_path, rebuild=True) == 0
    assert order[0] == "sync"
    assert order[1] == "ssh"
    assert order[2] == "push"
    assert [item for item in order if item.startswith("colmap")] == ["colmap1", "colmap2", "colmap3"]
    assert [item for item in order if item.startswith("splat")] == ["splat1", "splat2", "splat3"]
    assert config.is_file()
    assert not ply.is_file()
    script = scripts[0]
    assert "colmap" in script
    assert "outputs" not in script
    assert "/raw" not in script


def test_fresh_without_a_host_does_not_delete_colmap(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    _plant(tmp_path)
    ply = tmp_path / "data" / "can" / "colmap" / "orbit1" / "splat.ply"
    monkeypatch.setattr("tools.pipeline.running_vast_host", lambda: None)
    assert _skill_ingest("can", tmp_path, rebuild=True) == 2
    assert ply.is_file()


def test_run_step_uses_the_prompt_phrase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    launched: dict[str, str] = {}

    def fake_launch(_obj, _pairs, prompt=None, log_stem=None) -> str:
        del log_stem
        launched["prompt"] = str(prompt)
        return "echo STARTED 9\n"

    host = VastHost(instance_id=1, status="running", host="203.0.113.10", port=2222)
    monkeypatch.setattr("tools.pipeline.running_vast_host", lambda: host)
    monkeypatch.setattr("tools.block_b_job.host_launch_script", fake_launch)
    monkeypatch.setattr("tools.agent_loop._sync_host_code", lambda _host: 0)
    monkeypatch.setattr(
        "tools.agent_loop._splat_configs",
        lambda _obj, _root, _host: [("orbit1", "/root/SceneagentX/outputs/can_orbit1/splatfacto/t/config.yml")],
    )
    monkeypatch.setattr("tools.agent_loop._ssh", lambda _host, _script, timeout=90: (0, "STARTED 9\n", ""))
    assert _skill_run_step("can", tmp_path, PHRASE) == 0
    assert launched["prompt"] == PHRASE
    assert launched["prompt"] != CAN_CAPTION
