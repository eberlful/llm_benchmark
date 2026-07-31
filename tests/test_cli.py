import os
import pytest
import numpy as np
import yaml
import torch
from typer.testing import CliRunner
from src.cli import app

runner = CliRunner()

@pytest.fixture
def dummy_dataset_and_config(tmp_path):
    # Create dummy dataset directory
    dataset_dir = tmp_path / "dummy_dataset"
    dataset_dir.mkdir()
    
    # Create tiny train.bin and val.bin
    train_ids = np.random.randint(0, 1000, size=100, dtype=np.uint16)
    val_ids = np.random.randint(0, 1000, size=50, dtype=np.uint16)
    train_ids.tofile(dataset_dir / "train.bin")
    val_ids.tofile(dataset_dir / "val.bin")
    
    # Create configuration dict
    config = {
        "model": {
            "block_size": 8,
            "vocab_size": 50304,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "dropout": 0.0,
            "bias": False
        },
        "dataset": {
            "data_dir": str(dataset_dir)
        },
        "optimizer": {
            "weight_decay": 0.01,
            "learning_rate": 1e-4,
            "betas": [0.9, 0.99]
        },
        "trainer": {
            "max_iters": 2,
            "batch_size": 2,
            "block_size": 4,
            "learning_rate": 1e-4,
            "device": "cpu",
            "dtype": "float32",
            "eval_interval": 2,
            "eval_iters": 2,
            "decay_lr": False,
        }
    }
    
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    return config_file, dataset_dir

def test_cli_flow(dummy_dataset_and_config, tmp_path, monkeypatch):
    config_file, dataset_dir = dummy_dataset_and_config
    
    # Change current working directory to tmp_path using monkeypatch
    # so that the 'runs' directory is created inside the temp test folder.
    monkeypatch.chdir(tmp_path)
    
    # Run the train command (5 iterations)
    result = runner.invoke(app, ["train", str(config_file), "--steps", "5", "--batch-size", "1"])
    assert result.exit_code == 0, f"train command failed: {result.stdout}"
    
    # Assert runs/ folder exists and has at least one run
    runs_dir = tmp_path / "runs"
    assert runs_dir.exists()
    runs = os.listdir(runs_dir)
    assert len(runs) > 0
    
    run_dir = runs_dir / runs[0]
    
    # Assert single best checkpoint exists
    step_ckpts = list(run_dir.glob("ckpt_step_*.pt"))
    assert len(step_ckpts) == 1
    best_ckpt = step_ckpts[0]
    
    # Assert out.log exists
    log_file = run_dir / "out.log"
    assert log_file.exists()
    assert log_file.stat().st_size > 0

    # Assert config.yaml exists in run directory and contains overridden settings
    config_saved = run_dir / "config.yaml"
    assert config_saved.exists()
    with open(config_saved, "r", encoding="utf-8") as f:
        saved_data = yaml.safe_load(f)
    assert saved_data["trainer"]["max_iters"] == 5
    assert saved_data["trainer"]["batch_size"] == 1
    
    # Assert TensorBoard events exist
    tb_files = [f for f in os.listdir(run_dir) if "tfevents" in f]
    assert len(tb_files) > 0, "No TensorBoard event files found."
    for tb_file in tb_files:
        assert os.path.getsize(run_dir / tb_file) > 0
    
    # Run the eval command
    result_eval = runner.invoke(app, ["eval", "-c", str(best_ckpt), "-d", str(dataset_dir), "-b", "1", "-e", "2"])
    assert result_eval.exit_code == 0, f"eval command failed: {result_eval.stdout}"
    assert "Average Validation Loss" in result_eval.stdout
    
    # Run the inference command
    result_inf = runner.invoke(app, ["inference", "-c", str(best_ckpt), "-p", "Test prompt", "-n", "10"])
    assert result_inf.exit_code == 0, f"inference command failed: {result_inf.stdout}"
    assert "Generated Text Output" in result_inf.stdout

    # Run the inference command with prompt file
    prompt_file = tmp_path / "prompt.txt"
    prompt_file.write_text("Hello prompt file")
    result_inf_file = runner.invoke(app, ["inference", "-c", str(best_ckpt), "-f", str(prompt_file), "-n", "10"])
    assert result_inf_file.exit_code == 0, f"inference file command failed: {result_inf_file.stdout}"
    assert "Generated Text Output" in result_inf_file.stdout


def test_cli_flow_pam(tmp_path, monkeypatch):
    # Create dummy dataset directory
    dataset_dir = tmp_path / "dummy_dataset"
    dataset_dir.mkdir()
    
    # Create tiny train.bin and val.bin
    train_ids = np.random.randint(0, 1000, size=100, dtype=np.uint16)
    val_ids = np.random.randint(0, 1000, size=50, dtype=np.uint16)
    train_ids.tofile(dataset_dir / "train.bin")
    val_ids.tofile(dataset_dir / "val.bin")
    
    # Create configuration dict for PAM
    config = {
        "model": {
            "type": "pam",
            "block_size": 8,
            "vocab_size": 50304,
            "n_layer": 1,
            "n_head": 1,
            "n_embd": 8,
            "head_dim": 8,
            "dropout": 0.0,
            "bias": False
        },
        "dataset": {
            "data_dir": str(dataset_dir)
        },
        "optimizer": {
            "weight_decay": 0.01,
            "learning_rate": 1e-4,
            "betas": [0.9, 0.99]
        },
        "trainer": {
            "max_iters": 2,
            "batch_size": 2,
            "block_size": 4,
            "learning_rate": 1e-4,
            "device": "cpu",
            "dtype": "float32",
            "eval_interval": 2,
            "eval_iters": 2,
            "decay_lr": False,
        }
    }
    
    config_file = tmp_path / "config_pam.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
        
    monkeypatch.chdir(tmp_path)
    
    # Run the train command (5 iterations)
    result = runner.invoke(app, ["train", str(config_file), "--steps", "5", "--batch-size", "1"])
    assert result.exit_code == 0, f"train command failed: {result.stdout}"
    
    # Assert runs/ folder exists and has at least one run
    runs_dir = tmp_path / "runs"
    assert runs_dir.exists()
    runs = os.listdir(runs_dir)
    assert len(runs) > 0
    
    run_dir = runs_dir / runs[0]
    
    # Assert single best checkpoint exists
    step_ckpts = list(run_dir.glob("ckpt_step_*.pt"))
    assert len(step_ckpts) == 1
    best_ckpt = step_ckpts[0]
    
    # Assert config.yaml exists
    assert (run_dir / "config.yaml").exists()
    
    # Run the eval command
    result_eval = runner.invoke(app, ["eval", "-c", str(best_ckpt), "-d", str(dataset_dir), "-b", "1", "-e", "2"])
    assert result_eval.exit_code == 0, f"eval command failed: {result_eval.stdout}"
    assert "Average Validation Loss" in result_eval.stdout
    
    # Run the inference command
    result_inf = runner.invoke(app, ["inference", "-c", str(best_ckpt), "-p", "Test prompt", "-n", "10"])
    assert result_inf.exit_code == 0, f"inference command failed: {result_inf.stdout}"
    assert "Generated Text Output" in result_inf.stdout

    # Test resuming training from the checkpoint run directory
    result_resume = runner.invoke(app, ["train", str(config_file), "--steps", "10", "--resume", str(run_dir)])
    assert result_resume.exit_code == 0, f"resume training failed: {result_resume.stdout}"
    assert "Resuming training from checkpoint" in result_resume.stdout


def test_cli_compile_flag(dummy_dataset_and_config, tmp_path, monkeypatch):
    config_file, _ = dummy_dataset_and_config
    monkeypatch.chdir(tmp_path)

    # Run train with --no-compile
    result_no_compile = runner.invoke(app, ["train", str(config_file), "--steps", "1", "--no-compile"])
    assert result_no_compile.exit_code == 0, f"--no-compile failed: {result_no_compile.stdout}"
    assert "compiling the model" not in result_no_compile.stdout

    # Run train with --compile
    result_compile = runner.invoke(app, ["train", str(config_file), "--steps", "1", "--compile"])
    assert result_compile.exit_code == 0, f"--compile failed: {result_compile.stdout}"
    assert "compiling the model" in result_compile.stdout


def test_cli_create_xlstm_model():
    from src.cli import create_model_from_dict
    from src.models.xlstm import xLSTMModel

    cfg = {
        "type": "xlstm",
        "n_layer": 2,
        "n_head": 2,
        "n_embd": 32,
        "block_size": 16,
        "vocab_size": 100,
        "block_type_pattern": "7:1",
    }
    model = create_model_from_dict(cfg)
    assert isinstance(model, xLSTMModel)
    assert model.config.num_heads == 2
    assert model.config.n_embd == 32




