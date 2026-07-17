# 0003. Model Inheritance from nn.Module

We decided that all custom model architectures will inherit directly from `torch.nn.Module` (via our base `Model` class) and expose unified interfaces for forward computation (returning logits and loss), text generation, and optimizer configuration. This conforms to PyTorch patterns while establishing a consistent contract for different model backends.
