import os
import time
import math
from contextlib import nullcontext
from typing import List, Dict, Any, Optional, Tuple
import torch
from src.base.model import Model
from src.base.dataset import Dataset
from src.base.callback import Callback

class Trainer:
    """
    Core training loop orchestrator that feeds data batches to the Model,
    computes loss, backpropagates gradients, and updates weights.
    """

    def __init__(
        self,
        model: Model,
        dataset: Dataset,
        optimizer: torch.optim.Optimizer,
        max_iters: int,
        batch_size: int,
        block_size: int,
        learning_rate: float,
        decay_lr: bool = True,
        warmup_iters: int = 2000,
        lr_decay_iters: int = 600000,
        min_lr: float = 6e-5,
        grad_clip: float = 1.0,
        gradient_accumulation_steps: int = 1,
        device: str = 'cpu',
        dtype: str = 'float32',
        eval_interval: int = 2000,
        eval_iters: int = 200,
        out_dir: str = 'out',
        callbacks: Optional[List[Callback]] = None,
    ):
        self.model = model
        self.dataset = dataset
        self.optimizer = optimizer
        self.max_iters = max_iters
        self.batch_size = batch_size
        self.block_size = block_size
        self.learning_rate = learning_rate
        self.decay_lr = decay_lr
        self.warmup_iters = warmup_iters
        self.lr_decay_iters = lr_decay_iters
        self.min_lr = min_lr
        self.grad_clip = grad_clip
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.device = device
        self.dtype = dtype
        self.eval_interval = eval_interval
        self.eval_iters = eval_iters
        self.out_dir = out_dir
        self.callbacks = callbacks if callbacks is not None else []

        self.device_type = 'cuda' if 'cuda' in self.device else 'cpu'
        # Float16 data type will automatically use a GradScaler
        self.scaler = torch.amp.GradScaler('cuda', enabled=(self.dtype == 'float16'))
        ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[self.dtype]
        self.ctx = nullcontext() if self.device_type == 'cpu' else torch.amp.autocast(device_type=self.device_type, dtype=ptdtype)

        self.run_state: Dict[str, Any] = {
            'model': self.model,
            'optimizer': self.optimizer,
            'iter_num': 0,
            'max_iters': self.max_iters,
            'step_loss': 0.0,
            'step_time': 0.0,
            'lr': self.learning_rate,
            'mfu': -1.0,
            'train_loss': 0.0,
            'val_loss': 0.0,
            'out_dir': self.out_dir,
        }

    def _trigger_callbacks(self, hook_name: str) -> None:
        for callback in self.callbacks:
            hook = getattr(callback, hook_name, None)
            if hook:
                hook(self.run_state)

    def get_lr(self, it: int) -> float:
        # 1) linear warmup for warmup_iters steps
        if it < self.warmup_iters:
            return self.learning_rate * (it + 1) / (self.warmup_iters + 1)
        # 2) if it > lr_decay_iters, return min learning rate
        if it > self.lr_decay_iters:
            return self.min_lr
        # 3) in between, use cosine decay down to min learning rate
        if self.lr_decay_iters <= self.warmup_iters:
            return self.min_lr
        decay_ratio = (it - self.warmup_iters) / (self.lr_decay_iters - self.warmup_iters)
        decay_ratio = min(max(decay_ratio, 0.0), 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return self.min_lr + coeff * (self.learning_rate - self.min_lr)

    @torch.no_grad()
    def evaluate(self) -> Tuple[float, float]:
        self._trigger_callbacks('on_eval_start')
        self.model.eval()
        losses = {}
        for split in ['train', 'val']:
            split_losses = torch.zeros(self.eval_iters)
            for k in range(self.eval_iters):
                X, Y = self.dataset.get_batch(split, self.batch_size, self.block_size, self.device)
                with self.ctx:
                    _, loss = self.model(X, Y)
                split_losses[k] = loss.item()
            losses[split] = split_losses.mean().item()
        self.model.train()
        self.run_state['train_loss'] = losses['train']
        self.run_state['val_loss'] = losses['val']
        self._trigger_callbacks('on_eval_end')
        return losses['train'], losses['val']

    def train(self) -> None:
        self._trigger_callbacks('on_train_start')
        os.makedirs(self.out_dir, exist_ok=True)

        # Prepare the first batch
        X, Y = self.dataset.get_batch('train', self.batch_size, self.block_size, self.device)
        t0 = time.time()
        local_iter_num = 0
        running_mfu = -1.0

        iter_num = self.run_state['iter_num']

        while iter_num < self.max_iters:
            self._trigger_callbacks('on_step_start')

            # Determine and set the learning rate
            lr = self.get_lr(iter_num) if self.decay_lr else self.learning_rate
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.run_state['lr'] = lr

            # Evaluate loss periodic
            if iter_num % self.eval_interval == 0:
                self.evaluate()

            # Forward-backward-update
            step_loss_val = 0.0
            for micro_step in range(self.gradient_accumulation_steps):
                with self.ctx:
                    _, loss = self.model(X, Y)
                    scaled_loss = loss / self.gradient_accumulation_steps
                # Fetch next batch
                X, Y = self.dataset.get_batch('train', self.batch_size, self.block_size, self.device)
                self.scaler.scale(scaled_loss).backward()
                step_loss_val += loss.item()

            # Clip gradient
            if self.grad_clip != 0.0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

            # Step the optimizer
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)

            # Timings
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            # Estimate MFU
            if local_iter_num >= 5:
                # Unwrap model to get mfu estimation if method exists
                raw_model = self.model.module if hasattr(self.model, 'module') else self.model
                if hasattr(raw_model, 'estimate_mfu'):
                    # In old/train.py, the first arg is batch_size * gradient_accumulation_steps
                    mfu = raw_model.estimate_mfu(self.batch_size * self.gradient_accumulation_steps, dt)
                    running_mfu = mfu if running_mfu == -1.0 else 0.9 * running_mfu + 0.1 * mfu

            self.run_state['step_loss'] = step_loss_val / self.gradient_accumulation_steps
            self.run_state['step_time'] = dt
            self.run_state['mfu'] = running_mfu

            self._trigger_callbacks('on_step_end')

            iter_num += 1
            local_iter_num += 1
            self.run_state['iter_num'] = iter_num

        # Final eval at the end of training (if not already evaluated in the last step)
        if iter_num % self.eval_interval != 0:
            self.evaluate()
            
        self._trigger_callbacks('on_train_end')
