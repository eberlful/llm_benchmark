# 0007. Manual Data Preparation Requirement

We decided to require manual preparation of dataset files before running training or evaluation. If the required binary dataset files (`train.bin`, `val.bin`) are missing, the dataset loader will raise an error rather than attempting to download and encode the data automatically. This keeps the dataset components lightweight and free of network I/O responsibilities.
