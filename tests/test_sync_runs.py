import os
import shutil
import pytest
from pathlib import Path
from typer.testing import CliRunner

from scripts.sync_runs import (
    app as sync_app,
    get_run_info,
    list_run_dirs,
    resolve_run_path,
    copy_sync_directory,
)
from src.cli import app as main_app

runner = CliRunner()


@pytest.fixture
def run_env(tmp_path):
    """Fixture creating local runs directory and fake drive runs directory."""
    local_dir = tmp_path / "runs"
    drive_dir = tmp_path / "drive_runs"
    local_dir.mkdir()
    drive_dir.mkdir()

    # Create dummy run 1 locally
    run1 = local_dir / "run_20260101_100000"
    run1.mkdir()
    (run1 / "ckpt_step_100_val_loss_2.5000.pt").write_text("dummy checkpoint 1")
    (run1 / "config.yaml").write_text("dummy config 1")

    # Create dummy run 2 locally (newer timestamp)
    run2 = local_dir / "run_20260102_100000"
    run2.mkdir()
    (run2 / "ckpt_step_200_val_loss_2.1000.pt").write_text("dummy checkpoint 2")

    # Create dummy run on drive
    drive_run = drive_dir / "run_20260101_100000"
    drive_run.mkdir()
    (drive_run / "ckpt_step_100_val_loss_2.5000.pt").write_text("dummy checkpoint drive")

    return {
        "tmp_path": tmp_path,
        "local_dir": local_dir,
        "drive_dir": drive_dir,
        "run1": run1,
        "run2": run2,
        "drive_run": drive_run,
    }


def test_get_run_info(run_env):
    info = get_run_info(run_env["run1"])
    assert info["name"] == "run_20260101_100000"
    assert info["latest_step"] == 100
    assert info["best_val_loss"] == 2.5
    assert len(info["checkpoints"]) == 1


def test_list_run_dirs(run_env):
    dirs = list_run_dirs(run_env["local_dir"])
    assert len(dirs) == 2
    # run2 was created after run1 so it should be listed first (newest)
    assert dirs[0].name == "run_20260102_100000"


def test_resolve_run_path(run_env):
    path, name = resolve_run_path(None, run_env["local_dir"], latest=True)
    assert name == "run_20260102_100000"

    path, name = resolve_run_path("run_20260101_100000", run_env["local_dir"])
    assert name == "run_20260101_100000"


def test_upload_cmd_latest(run_env):
    result = runner.invoke(
        sync_app,
        ["upload", "--latest", "--local-dir", str(run_env["local_dir"]), "--drive-dir", str(run_env["drive_dir"]), "--no-mount"]
    )
    assert result.exit_code == 0
    uploaded_run2 = run_env["drive_dir"] / "run_20260102_100000"
    assert uploaded_run2.exists()
    assert (uploaded_run2 / "ckpt_step_200_val_loss_2.1000.pt").exists()


def test_upload_cmd_all(run_env):
    result = runner.invoke(
        sync_app,
        ["upload", "--all", "--local-dir", str(run_env["local_dir"]), "--drive-dir", str(run_env["drive_dir"]), "--no-mount"]
    )
    assert result.exit_code == 0
    assert (run_env["drive_dir"] / "run_20260101_100000").exists()
    assert (run_env["drive_dir"] / "run_20260102_100000").exists()


def test_download_cmd_latest(run_env):
    # Place a new run on drive
    new_drive_run = run_env["drive_dir"] / "run_20260103_120000"
    new_drive_run.mkdir()
    (new_drive_run / "last_ckpt.pt").write_text("drive checkpoint 3")

    result = runner.invoke(
        sync_app,
        ["download", "--latest", "--local-dir", str(run_env["local_dir"]), "--drive-dir", str(run_env["drive_dir"]), "--no-mount"]
    )
    assert result.exit_code == 0
    local_downloaded = run_env["local_dir"] / "run_20260103_120000"
    assert local_downloaded.exists()
    assert (local_downloaded / "last_ckpt.pt").exists()


def test_list_cmd(run_env):
    result = runner.invoke(
        sync_app,
        ["list", "--local-dir", str(run_env["local_dir"]), "--drive-dir", str(run_env["drive_dir"]), "--no-mount"]
    )
    assert result.exit_code == 0
    assert "Training Runs Status" in result.output
    assert "run_20260101_100000" in result.output


def test_main_cli_sync(run_env):
    # Verify main app contains 'sync' subcommand
    result = runner.invoke(
        main_app,
        ["sync", "list", "--local-dir", str(run_env["local_dir"]), "--drive-dir", str(run_env["drive_dir"]), "--no-mount"]
    )
    assert result.exit_code == 0
    assert "Training Runs Status" in result.output
