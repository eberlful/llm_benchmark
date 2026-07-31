# Model Training Environment

A training and evaluation platform for various language model architectures.

## Language

**Configuration**:
The specification of hyperparameters, data paths, logging configs, and training parameters for a run.
_Avoid_: Params, settings

**Training Run**:
A single execution of the training process, outputting checkpoints, TensorBoard logs, and terminal logs to a dedicated directory.
_Avoid_: Job, execution, trial

**Trainer**:
The core engine orchestrating the training loop, feeding data batches to the Model, computing loss, backpropagating gradients, updating weights, and triggering Callbacks.
_Avoid_: Runner, Engine

**Model**:
A generic base class wrapping the neural network architecture (e.g., GPT), responsible for the forward pass, computing logits/loss, and loading/saving weights.
_Avoid_: Network, Architecture

**Dataset**:
A generic base class wrapping data preparation and generation of batches (e.g., train/val splits).
_Avoid_: DataLoader, DataProvider

**Logger**:
A base class for recording metrics and messages during training (e.g., TerminalLogger, TensorBoardLogger).
_Avoid_: Writer, MetricTracker

**Callback**:
A base class for hooks executed at specific points during a Training Run (e.g., on epoch start/end, on step start/end) to perform auxiliary tasks like checkpointing or custom evaluation.
_Avoid_: Hook, Plugin

**Phase-Associative Memory (PAM)**:
A complex-valued sequence modeling architecture whose internal state is a complex matrix $S_t \in \mathbb{C}^{d \times d}$ representing content-addressable associative storage. It retrieves values using the complex-conjugate inner product between keys and queries, achieving selective retrieval via constructive/destructive interference without softmax attention.
_Avoid_: Attention, Attention-free SSM

**Complex Gated Unit (CGU)**:
A channel-mixing module in complex-valued neural networks that gates the signal using magnitude and phase, serving as the feed-forward / channel-mixing block in the PAM model.
_Avoid_: FeedForward, MLP

**Complex Representation (Split-Real Form)**:
Representing complex numbers $z = a + ib$ using a float tensor with a final dimension of size 2 (e.g., shape `[..., d, 2]`).
_Avoid_: Complex dtype, Real projection

**Extended LSTM (xLSTM)**:
A sequence modeling architecture built from residual stacks of mLSTM and sLSTM blocks featuring exponential gating and stabilized state updates.
_Avoid_: Modern LSTM, Custom Recurrent Net

**Matrix LSTM (mLSTM)**:
An xLSTM block variant with a matrix memory state updated via a key-value covariance rule, enabling fully parallel causal training and step-by-step recurrent inference.
_Avoid_: Matrix Recurrent Cell, Linear Attention Layer

**Scalar LSTM (sLSTM)**:
An xLSTM block variant with scalar memory cells, exponential gating, numerical max-stabilization state tracking, and multi-head memory mixing.
_Avoid_: Standard LSTM, Stabilized Recurrent Block

**Parallel Causal Formulation**:
A sequence-level vectorized forward pass for mLSTM training that computes matrix memory outputs across sequence length $T$ using lower-triangular causal decay matrices and log-sum-exp stabilization.
_Avoid_: Recurrent Loop Training, Unrolled Step Training

**Recurrent Step Formulation**:
A token-by-token state update pass for xLSTM inference (in `generate`), maintaining state vectors/matrices across time steps $t$.
_Avoid_: Full Sequence Re-computation, KV-Cache Attention

**Block Pattern Specification**:
A configurable ordering rule defining the sequence of mLSTM and sLSTM residual blocks across model layers (e.g. `"mlstm"`, `"slstm"`, `"7:1"`, or an explicit list `["mlstm", "mlstm", "slstm"]`).
_Avoid_: Layer Type Parameter, Static Architecture Layout





