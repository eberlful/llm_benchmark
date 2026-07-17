import os
import urllib.request
import numpy as np
import torch
import tiktoken
from typing import Tuple
from src.base.dataset import Dataset

class ShakespeareDataset(Dataset):
    def __init__(self, data_dir: str = "data/shakespeare"):
        self.data_dir = data_dir
        self.train_bin_path = os.path.join(data_dir, "train.bin")
        self.val_bin_path = os.path.join(data_dir, "val.bin")
        
        # Check if the processed files exist. If not, download and prepare.
        if not (os.path.exists(self.train_bin_path) and os.path.exists(self.val_bin_path)):
            self._download_and_prepare()

    def _download_and_prepare(self) -> None:
        print(f"Dataset binary files not found. Preparing Shakespeare dataset in {self.data_dir}...")
        os.makedirs(self.data_dir, exist_ok=True)
        
        # Download input.txt
        input_file_path = os.path.join(self.data_dir, "input.txt")
        if not os.path.exists(input_file_path):
            url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
            print(f"Downloading source text from {url}...")
            urllib.request.urlretrieve(url, input_file_path)
            
        print("Tokenizing input text...")
        with open(input_file_path, "r", encoding="utf-8") as f:
            data = f.read()
            
        n = len(data)
        train_data = data[:int(n * 0.9)]
        val_data = data[int(n * 0.9):]
        
        enc = tiktoken.get_encoding("gpt2")
        train_ids = enc.encode_ordinary(train_data)
        val_ids = enc.encode_ordinary(val_data)
        
        print(f"Train set has {len(train_ids):,} tokens")
        print(f"Val set has {len(val_ids):,} tokens")
        
        train_ids_np = np.array(train_ids, dtype=np.uint16)
        val_ids_np = np.array(val_ids, dtype=np.uint16)
        
        train_ids_np.tofile(self.train_bin_path)
        val_ids_np.tofile(self.val_bin_path)
        print("Preparation complete.")

    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        bin_path = self.train_bin_path if split == "train" else self.val_bin_path
        if not os.path.exists(bin_path):
            raise FileNotFoundError(f"Binary file {bin_path} does not exist. Call __init__ first.")
            
        # Recreate np.memmap each time to avoid memory leak
        data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        
        if len(data) <= block_size:
            raise ValueError(f"Dataset split size {len(data)} is too small for block_size {block_size}.")
            
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
        
        if "cuda" in device:
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
            
        return x, y
