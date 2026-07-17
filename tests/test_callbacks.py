import os
import shutil
import pytest
import torch
from src.datasets.shakespeare import ShakespeareDataset
from src.models.gpt import GPTModel, GPTConfig
from src.callbacks.tensorboard_logger import TensorBoardLogger
from src.callbacks.checkpoint import CheckpointCallback
from src.trainer import Trainer

@pytest.fixture
def temp_dataset_dir(tmp_path):
    d = tmp_path / "shakespeare"
    yield str(d)
    if d.exists():
        shutil.rmtree(d)

@pytest.fixture
def temp_run_dir(tmp_path):
    d = tmp_path / "run_out"
    yield str(d)
    if d.exists():
        shutil.rmtree(d)

def test_tensorboard_logger_and_checkpoint_callbacks(temp_dataset_dir, temp_run_dir):
    # Initialize Shakespeare Dataset
    dataset = ShakespeareDataset(data_dir=temp_dataset_dir)
    
    # Initialize a tiny GPT model
    config = GPTConfig(
        block_size=16,
        vocab_size=50304,
        n_layer=1,
        n_head=1,
        n_embd=16,
        dropout=0.0,
        bias=False
    )
    model = GPTModel(config)
    
    # Configure optimizer
    optimizer = model.configure_optimizers(
        weight_decay=0.01,
        learning_rate=1e-4,
        betas=(0.9, 0.99),
        device_type="cpu"
    )
    
    # Initialize callbacks
    tb_logger = TensorBoardLogger(log_interval=1)
    checkpoint_callback = CheckpointCallback()
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        max_iters=4,
        batch_size=2,
        block_size=8,
        learning_rate=1e-4,
        decay_lr=True,
        warmup_iters=2,
        lr_decay_iters=4,
        min_lr=1e-5,
        grad_clip=1.0,
        gradient_accumulation_steps=1,
        device="cpu",
        dtype="float32",
        eval_interval=2,
        eval_iters=2,
        out_dir=temp_run_dir,
        callbacks=[tb_logger, checkpoint_callback]
    )
    
    # Run training
    trainer.train()
    
    # 1. Assert TensorBoard event files are created
    tb_files = [f for f in os.listdir(temp_run_dir) if "tfevents" in f]
    assert len(tb_files) > 0, "No TensorBoard event files generated."
    # Ensure they have non-zero size
    for tb_file in tb_files:
        assert os.path.getsize(os.path.join(temp_run_dir, tb_file)) > 0
    
    # 2. Assert Checkpoints are created
    best_ckpt_path = os.path.join(temp_run_dir, "best_ckpt.pt")
    last_ckpt_path = os.path.join(temp_run_dir, "last_ckpt.pt")
    
    assert os.path.exists(best_ckpt_path), "best_ckpt.pt not found."
    assert os.path.exists(last_ckpt_path), "last_ckpt.pt not found."
    
    # Load and verify checkpoints
    best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
    last_ckpt = torch.load(last_ckpt_path, map_location="cpu", weights_only=False)
    
    for ckpt in [best_ckpt, last_ckpt]:
        assert "model" in ckpt
        assert "optimizer" in ckpt
        assert "config" in ckpt
        assert "steps" in ckpt
        assert "best_val_loss" in ckpt
        
        # Verify weight matrices match or exist
        assert isinstance(ckpt["model"], dict)
        assert "transformer.wte.weight" in ckpt["model"]
        
        # Verify config is correct
        assert ckpt["config"].n_embd == 16
        assert ckpt["config"].n_layer == 1
        
        # Verify optimizer dictionary is correct
        assert isinstance(ckpt["optimizer"], dict)
        
        # Verify steps is valid
        assert isinstance(ckpt["steps"], int)
        assert ckpt["steps"] >= 0
        
        # Verify best_val_loss is valid
        assert isinstance(ckpt["best_val_loss"], float)
        assert ckpt["best_val_loss"] > 0.0
