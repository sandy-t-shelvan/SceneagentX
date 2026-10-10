from tools.vast_host import current_host
from tools.video_input import find_video, record_absent


def test_current_host_prefers_the_running_instance():
    rows = [
        {"id": 1, "actual_status": "exited", "public_ipaddr": "1.1.1.1", "direct_port_start": 22},
        {"id": 2, "actual_status": "running", "public_ipaddr": "2.2.2.2", "direct_port_start": 6525},
    ]
    host = current_host(rows)
    assert host.instance_id == 2
    assert host.host == "2.2.2.2"
    assert host.port == 6525


def test_missing_video_does_not_invent_a_path(tmp_path):
    assert find_video("scan", tmp_path) is None
    path = record_absent("scan", tmp_path)
    text = path.read_text(encoding="utf-8")
    assert "not_started" in text
    assert "provided" in text


def test_pipeline_does_not_start_colmap_without_orbit_video(tmp_path, monkeypatch):
    called = {"colmap": False, "splat": False}

    def colmap(*args, **kwargs):
        called["colmap"] = True
        return 0

    def splat(*args, **kwargs):
        called["splat"] = True
        return 0

    monkeypatch.setattr("tools.pipeline.run_b0", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.check_object", lambda *args, **kwargs: 0)
    monkeypatch.setattr("tools.pipeline.run_colmap_orbit", colmap)
    monkeypatch.setattr("tools.pipeline.run_splat_orbit", splat)
    from tools.pipeline import run

    code = run("scan", through="B3", root=tmp_path)
    assert code == 2
    assert called == {"colmap": False, "splat": False}
    text = (tmp_path / "data" / "scan" / "video_path.json").read_text(encoding="utf-8")
    assert "not_started" in text
