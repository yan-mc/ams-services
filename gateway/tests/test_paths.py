from __future__ import annotations

from pathlib import Path

from app.config import Settings
from app.paths import materialize_file, sanitize_job_params


def make_settings(tmp_path: Path) -> Settings:
    settings = Settings(ams_home=tmp_path / "ams")
    settings.ensure_dirs()
    return settings


def test_external_input_is_copied_to_projects(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    external = tmp_path / "external.wav"
    external.write_bytes(b"wav")
    params = sanitize_job_params(
        {"input_path": str(external), "voice_name": "曼波"},
        settings=settings,
        project_id="p1",
    )
    copied = Path(params["input_path"])
    assert copied.is_file()
    assert copied.is_relative_to(settings.ams_home)
    assert copied.read_bytes() == b"wav"
    assert params["voice_name"] == "曼波"
    assert params["_ams_imports"][0]["from"] == str(external)


def test_data_input_is_not_copied(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    inside = settings.projects_dir / "inside.wav"
    inside.write_bytes(b"inside")
    params = sanitize_job_params({"input_path": str(inside)}, settings=settings, project_id=None)
    assert params["input_path"] == str(inside)
    assert "_ams_imports" not in params


def test_external_output_is_redirected_into_projects(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    external_out = tmp_path / "external-out"
    external_out.mkdir()
    params = sanitize_job_params(
        {"input_path": None, "out_dir": str(external_out)},
        settings=settings,
        project_id="p1",
    )
    target = Path(params["out_dir"])
    assert target.is_dir()
    assert target.is_relative_to(settings.ams_home)


def test_materialize_file_returns_data_path(tmp_path: Path) -> None:
    settings = make_settings(tmp_path)
    external = tmp_path / "asset.pth"
    external.write_bytes(b"pth")
    target = Path(materialize_file(str(external), settings=settings, project_id="p2"))
    assert target.is_file()
    assert target.is_relative_to(settings.ams_home)