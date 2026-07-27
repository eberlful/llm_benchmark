import os
import torch
from typing import Dict, Any, Optional
from rich.console import Console
from src.base.callback import Callback

class CheckpointCallback(Callback):
    """
    Callback that saves only the single best model checkpoint during training,
    formatted with step count and validation loss in the filename.
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

        is_best = val_loss < self.best_val_loss
        if not is_best:
            return

        self.best_val_loss = val_loss

        model = run_state['model']
        optimizer = run_state['optimizer']
        raw_model = model.module if hasattr(model, 'module') else model
        raw_model = getattr(raw_model, '_orig_mod', raw_model)
        config = getattr(raw_model, 'config', None)
        steps = run_state['iter_num']

        checkpoint = {
            'model': raw_model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'config': config,
            'step': steps,
            'steps': steps,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss,
        }

        # Remove old checkpoint files in out_dir so only the best remains
        for fname in os.listdir(out_dir):
            if (fname.startswith("ckpt_step_") and fname.endswith(".pt")) or fname in ["last_ckpt.pt", "best_ckpt.pt"]:
                try:
                    os.remove(os.path.join(out_dir, fname))
                except OSError:
                    pass

        # Save single best checkpoint with step and monitor value in filename
        step_filename = f"ckpt_step_{steps}_val_loss_{val_loss:.4f}.pt"
        step_path = os.path.join(out_dir, step_filename)
        torch.save(checkpoint, step_path)
        self.console.print(f"[bold green]🏆 Saved new best checkpoint to [cyan]{step_path}[/cyan] (step {steps}, val_loss: {val_loss:.4f})[/bold green]")
