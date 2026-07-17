import os
from typing import Dict, Any, Optional
from torch.utils.tensorboard import SummaryWriter
from src.base.logger import Logger

class TensorBoardLogger(Logger):
    """
    Logger subclass that records metrics to TensorBoard via SummaryWriter.
    """

    def __init__(self, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval
        self.writer: Optional[SummaryWriter] = None

    def on_train_start(self, run_state: Dict[str, Any]) -> None:
        out_dir = run_state.get('out_dir', 'out')
        os.makedirs(out_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=out_dir)

    def on_step_end(self, run_state: Dict[str, Any]) -> None:
        iter_num = run_state['iter_num']
        if iter_num % self.log_interval == 0 and self.writer is not None:
            self.writer.add_scalar("train/step_loss", run_state['step_loss'], iter_num)
            self.writer.add_scalar("train/lr", run_state['lr'], iter_num)
            self.writer.add_scalar("train/step_time", run_state['step_time'], iter_num)
            
            mfu = run_state.get('mfu')
            if mfu is not None and mfu >= 0:
                self.writer.add_scalar("train/mfu", mfu, iter_num)

    def on_eval_end(self, run_state: Dict[str, Any]) -> None:
        if self.writer is not None:
            iter_num = run_state['iter_num']
            if 'train_loss' in run_state:
                self.writer.add_scalar("val/train_loss", run_state['train_loss'], iter_num)
            if 'val_loss' in run_state:
                self.writer.add_scalar("val/val_loss", run_state['val_loss'], iter_num)

    def on_train_end(self, run_state: Dict[str, Any]) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
