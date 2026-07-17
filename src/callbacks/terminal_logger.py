import os
from typing import Dict, Any, Optional
from rich.console import Console
from src.base.logger import Logger

class TerminalLogger(Logger):
    """
    Logger subclass that formats metric logs to the terminal with Rich and mirrors them to out.log.
    """

    def __init__(self, log_interval: int = 1):
        super().__init__()
        self.log_interval = log_interval
        self.console = Console()
        self.file_console: Optional[Console] = None
        self.file_handle = None

    def on_train_start(self, run_state: Dict[str, Any]) -> None:
        out_dir = run_state.get('out_dir', 'out')
        os.makedirs(out_dir, exist_ok=True)
        log_file_path = os.path.join(out_dir, "out.log")
        self.file_handle = open(log_file_path, "a", encoding="utf-8")
        self.file_console = Console(file=self.file_handle, force_terminal=False, color_system=None)
        
        msg = f"🚀 Starting training run. Output directory: {out_dir}"
        self.log(msg)

    def log(self, message: str) -> None:
        # Print to stdout console (rich formatted)
        self.console.print(message)
        # Print to file console (stripped of styles)
        if self.file_console:
            self.file_console.print(message)
            if self.file_handle:
                self.file_handle.flush()

    def on_step_end(self, run_state: Dict[str, Any]) -> None:
        iter_num = run_state['iter_num']
        max_iters = run_state['max_iters']
        if iter_num % self.log_interval == 0:
            loss = run_state['step_loss']
            lr = run_state['lr']
            dt = run_state['step_time']
            mfu = run_state['mfu']
            mfu_str = f"{mfu * 100:.2f}%" if mfu is not None and mfu >= 0 else "N/A"
            msg = f"📥 Step [bold cyan]{iter_num}[/bold cyan]/{max_iters} | 📉 Loss: [bold red]{loss:.4f}[/bold red] | ⚡ LR: [bold yellow]{lr:.2e}[/bold yellow] | ⏱️ Time: [bold green]{dt * 1000:.2f}ms[/bold green] | 📊 MFU: [bold blue]{mfu_str}[/bold blue]"
            self.log(msg)

    def on_eval_start(self, run_state: Dict[str, Any]) -> None:
        self.log("🔍 Running evaluation...")

    def on_eval_end(self, run_state: Dict[str, Any]) -> None:
        train_loss = run_state.get('train_loss', 0.0)
        val_loss = run_state.get('val_loss', 0.0)
        msg = f"🏆 Eval results | 📈 Train Loss: [bold green]{train_loss:.4f}[/bold green] | 📉 Val Loss: [bold red]{val_loss:.4f}[/bold red]"
        self.log(msg)

    def on_train_end(self, run_state: Dict[str, Any]) -> None:
        self.log("🏁 Training completed!")
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
            self.file_console = None
