import os
import shutil
import pytest
import torch
from src.datasets.shakespeare import ShakespeareDataset
from src.models.gpt import GPTModel, GPTConfig
from src.callbacks.terminal_logger import TerminalLogger
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

def test_trainer_integration(temp_dataset_dir, temp_run_dir):
    # Initialize Shakespeare Dataset
    dataset = ShakespeareDataset(data_dir=temp_dataset_dir)
    
    # Initialize a tiny GPT model to keep it fast on CPU
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
    
    # Initialize terminal logger with log_interval = 1
    logger = TerminalLogger(log_interval=1)
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        max_iters=5,
        batch_size=2,
        block_size=8,
        learning_rate=1e-4,
        decay_lr=True,
        warmup_iters=2,
        lr_decay_iters=5,
        min_lr=1e-5,
        grad_clip=1.0,
        gradient_accumulation_steps=2,
        device="cpu",
        dtype="float32",
        eval_interval=2,
        eval_iters=2,
        out_dir=temp_run_dir,
        callbacks=[logger]
    )
    
    # Run training
    trainer.train()
    
    # Assertions
    assert trainer.run_state['iter_num'] == 5
    
    # Check that out.log was created and contains expected content
    log_file_path = os.path.join(temp_run_dir, "out.log")
    assert os.path.exists(log_file_path)
    
    with open(log_file_path, "r", encoding="utf-8") as f:
        log_content = f.read()
        
    print("Log content:\n", log_content)
    
    assert "Starting training run" in log_content
    assert "Step 0" in log_content
    assert "Step 1" in log_content
    assert "Step 4" in log_content
    assert "Eval results" in log_content
    assert "Training completed!" in log_content
