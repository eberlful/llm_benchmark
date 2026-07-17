from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Tuple, Optional

class Model(nn.Module, ABC):
    """
    Abstract base class for all neural network models in the benchmark system.
    """
    
    def __init__(self) -> None:
        super().__init__()
    
    @abstractmethod
    def forward(
        self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of the model.
        
        Args:
            idx: Input token indices, shape (batch_size, sequence_length)
            targets: Optional target token indices, shape (batch_size, sequence_length)
            
        Returns:
            Tuple[torch.Tensor, Optional[torch.Tensor]]: A tuple of (logits, loss).
                logits: shape (batch_size, sequence_length, vocab_size) if targets is provided,
                        otherwise can be (batch_size, 1, vocab_size) or similar.
                loss: cross entropy loss tensor if targets is provided, else None.
        """
        pass

    @abstractmethod
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate sequence completion starting from the context sequence idx.
        
        Args:
            idx: Conditioning sequence indices, shape (batch_size, sequence_length)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher is more random)
            top_k: Only sample from the top k logits if specified
            
        Returns:
            torch.Tensor: Sequence indices including the generated tokens, shape (batch_size, sequence_length + max_new_tokens)
        """
        pass

    @abstractmethod
    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: Tuple[float, float],
        device_type: str,
    ) -> torch.optim.Optimizer:
        """
        Configure the optimizer for training.
        
        Args:
            weight_decay: Weight decay coefficient
            learning_rate: Learning rate
            betas: Betas tuple for AdamW optimizer
            device_type: Device type (e.g. 'cpu', 'cuda')
            
        Returns:
            torch.optim.Optimizer: Configured PyTorch optimizer
        """
        pass
