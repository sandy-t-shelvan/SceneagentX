import json

from tools.vast_search import create_command, main, select_offer

CHEAP_SHORT = {
    "id": 1,
    "gpu_name": "RTX 4090",
    "num_gpus": 1,
    "gpu_ram": 24.6,
    "cpu_cores_effective": 16,
    "cpu_ram": 64.0,
    "disk_space": 400,
    "direct_port_count": 8,
    "driver_version": "595.84",
    "dph_total": 0.30,
}
WASHINGTON = {
    "id": 53768455,
    "gpu_name": "RTX 4090",
    "num_gpus": 1,
    "gpu_ram": 24.6,
    "cpu_cores_effective": 32,
    "cpu_ram": 86.0,
    "disk_space": 2124,
    "direct_port_count": 66,
    "driver_version": "595.84",
    "dph_total": 0.4863,
}
OLD_DRIVER = {
    **WASHINGTON,
    "id": 9,
    "driver_version": "550.127.08",
    "dph_total": 0.20,
}


def test_selects_24_core_4090_with_driver_580():
    chosen = select_offer([CHEAP_SHORT, OLD_DRIVER, WASHINGTON])
    assert chosen is not None
    assert chosen["id"] == 53768455
    command = create_command(53768455)
    assert "--disk 250" in command
    assert "--ssh" in command
    assert "--direct" in command
    assert "pytorch/pytorch:2.1.2-cuda12.1-cudnn8-devel" in command
    assert "api-key" not in command
    assert "api_key" not in command


def test_cli_prints_create_command_from_fixture(tmp_path, capsys):
    fixture = tmp_path / "offers.json"
    fixture.write_text(json.dumps([CHEAP_SHORT, WASHINGTON]), encoding="utf-8")
    assert main(["--fixture", str(fixture)]) == 0
    out = capsys.readouterr().out
    assert "53768455" in out
    assert "This command is not run here" in out
    assert "api-key" not in out
