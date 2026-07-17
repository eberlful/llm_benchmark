# 0002. Dataset Batch-Provider Interface

We decided to design the `Dataset` base class with a custom batch-provider interface (`get_batch`) rather than forcing a standard PyTorch `DataLoader`. This matches the optimized, lightweight 1D sequence memory-mapped data loading strategy used in nanoGPT and avoids multi-process overhead, while remaining generic enough to support other datasets.
