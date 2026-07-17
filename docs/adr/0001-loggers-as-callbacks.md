# 0001. Loggers as Callbacks

We decided to model all Loggers as subclasses of the Callback base class. This allows the Trainer to manage loggers under a single, unified callback execution pipeline, simplifying the training loop design and making the logging architecture fully extensible.
