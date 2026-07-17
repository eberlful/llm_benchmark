from abc import ABC, abstractmethod
import torch
from typing import Tuple

class Dataset(ABC):
    @abstractmethod
    def get_batch(
        self, split: str, batch_size: int, block_size: int, device: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Retrieve a batch of input tokens X and target tokens Y for the given split.
        
        Args:
            split: 'train' or 'val'
            batch_size: the batch size
            block_size: the sequence length (context size)
            device: the target device for returned tensors (e.g. 'cpu', 'cuda')
            
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple (X, Y) of PyTorch tensors.
                X has shape (batch_size, block_size)
                Y has shape (batch_size, block_size)
        """
        pass
