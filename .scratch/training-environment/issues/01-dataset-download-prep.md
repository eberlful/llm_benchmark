# 01 — Shakespeare Dataset with Automated Download & Prep

**What to build:** 
A data loading system that automatically downloads and tokenizes the Shakespeare corpus if the processed binaries are missing. It must expose a generic `Dataset` base class and a concrete `ShakespeareDataset` implementing the `get_batch` contract to retrieve input and target token slices.

**Blocked by:** None — can start immediately

**Status:** completed

- [x] A generic `Dataset` base class is created at `src/base/dataset.py` with an abstract method `get_batch(split, batch_size, block_size, device)`.
- [x] A concrete `ShakespeareDataset` is created at `src/datasets/shakespeare.py` inheriting from `Dataset`.
- [x] The dataset checks for the existence of `train.bin` and `val.bin` under `data/shakespeare/`.
- [x] If missing, the dataset downloads `input.txt` from `https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt`, encodes it using `tiktoken` (GPT-2 encoding), and writes `train.bin` and `val.bin`.
- [x] The `get_batch(split, batch_size, block_size, device)` method loads token slices using `np.memmap` and returns `X, Y` as PyTorch tensors moved to the specified device.
- [x] A script or test is run to verify that the dataset automatically downloads, encodes, saves, and can successfully generate batches of the correct shape.
