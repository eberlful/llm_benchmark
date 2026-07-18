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


