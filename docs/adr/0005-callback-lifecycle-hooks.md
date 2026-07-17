# 0005. Callback Lifecycle Hooks

We decided to implement a step-centric and evaluation-focused lifecycle hook interface for our Callbacks: `on_train_start`, `on_train_end`, `on_step_start`, `on_step_end`, `on_eval_start`, and `on_eval_end`. This aligns with the step-based training dynamics of language models and avoids the complexity of full epoch-based schedules, which are less relevant for large-scale token datasets.
