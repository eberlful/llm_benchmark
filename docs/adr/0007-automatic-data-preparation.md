# 0007. Automatic Data Preparation

We decided to support automatic data preparation within our concrete `Dataset` implementations. If tokenized binary files are missing when loading a dataset, the system will download the source material and run the tokenization process automatically before resuming execution, removing manual setup friction.
