import os
import shutil
import pytest
import torch
from src.datasets.shakespeare import ShakespeareDataset

@pytest.fixture
def temp_data_dir(tmp_path):
    d = tmp_path / "shakespeare"
    yield str(d)
    if d.exists():
        shutil.rmtree(d)

def test_shakespeare_dataset_download_and_prep(temp_data_dir):
    # Instantiate the dataset, which should download, encode, and save train.bin/val.bin
    dataset = ShakespeareDataset(data_dir=temp_data_dir)
    
    # Check that files were created
    assert os.path.exists(dataset.train_bin_path)
    assert os.path.exists(dataset.val_bin_path)
    assert os.path.exists(os.path.join(temp_data_dir, "input.txt"))
    
    # Check that they have non-zero size
    assert os.path.getsize(dataset.train_bin_path) > 0
    assert os.path.getsize(dataset.val_bin_path) > 0

def test_shakespeare_dataset_get_batch(temp_data_dir):
    dataset = ShakespeareDataset(data_dir=temp_data_dir)
    
    batch_size = 4
    block_size = 8
    device = "cpu"
    
    x, y = dataset.get_batch(split="train", batch_size=batch_size, block_size=block_size, device=device)
    
    # Assert types and shapes
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
    
    # Check device
    assert x.device.type == "cpu"
    assert y.device.type == "cpu"
    
    # Check alignment: target should be shifted by 1 relative to input
    for r in range(batch_size):
        torch.testing.assert_close(y[r, :-1], x[r, 1:])

def test_shakespeare_dataset_val_split(temp_data_dir):
    dataset = ShakespeareDataset(data_dir=temp_data_dir)
    
    batch_size = 2
    block_size = 4
    device = "cpu"
    
    x, y = dataset.get_batch(split="val", batch_size=batch_size, block_size=block_size, device=device)
    assert x.shape == (batch_size, block_size)
    assert y.shape == (batch_size, block_size)
