import os
import torch
from typing import Dict, Any, Optional
from rich.console import Console
from src.base.callback import Callback

class CheckpointCallback(Callback):
    """
    Callback that saves best and last model checkpoints during training.
    """

    def __init__(self, console: Optional[Console] = None) -> None:
        super().__init__()
        self.best_val_loss = float('inf')
        self.console = console or Console()

    def on_train_start(self, run_state: Dict[str, Any]) -> None:
        # Reset best validation loss at start of training run
        self.best_val_loss = float('inf')

    def on_eval_end(self, run_state: Dict[str, Any]) -> None:
        val_loss = run_state.get('val_loss')
        if val_loss is None:
            return

        out_dir = run_state.get('out_dir', 'out')
        os.makedirs(out_dir, exist_ok=True)

        model = run_state['model']
        optimizer = run_state['optimizer']
        raw_model = model.module if hasattr(model, 'module') else model
        config = getattr(raw_model, 'config', None)
        steps = run_state['iter_num']

        is_best = val_loss < self.best_val_loss
        if is_best:
            self.best_val_loss = val_loss

        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'step': steps,
            'steps': steps,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }

        # Save last checkpoint
        last_path = os.path.join(out_dir, "last_ckpt.pt")
        torch.save(checkpoint, last_path)
        self.console.print(f"[bold green]💾 Saved last checkpoint to [cyan]{last_path}[/cyan] (step {steps}, val_loss: {val_loss:.4f})[/bold green]")

        # Save best checkpoint if validation loss improved
        if is_best:
            best_path = os.path.join(out_dir, "best_ckpt.pt")
            torch.save(checkpoint, best_path)
            self.console.print(f"[bold green]🏆 Saved new best checkpoint to [cyan]{best_path}[/cyan] (step {steps}, val_loss: {val_loss:.4f})[/bold green]")

