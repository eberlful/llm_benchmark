from abc import ABC
from typing import Dict, Any

class Callback(ABC):
    """
    Abstract base class for all training lifecycle hooks (callbacks).
    """

    def on_train_start(self, run_state: Dict[str, Any]) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, run_state: Dict[str, Any]) -> None:
        """Called at the end of training."""
        pass

    def on_step_start(self, run_state: Dict[str, Any]) -> None:
        """Called at the beginning of a training step."""
        pass

    def on_step_end(self, run_state: Dict[str, Any]) -> None:
        """Called at the end of a training step."""
        pass

    def on_eval_start(self, run_state: Dict[str, Any]) -> None:
        """Called at the beginning of an evaluation run."""
        pass

    def on_eval_end(self, run_state: Dict[str, Any]) -> None:
        """Called at the end of an evaluation run."""
        pass
