# 0004. Modular Package Structure

We decided to structure the training environment code inside a modular `src/` directory. This splits the core generic framework (base classes and trainer) from concrete model architectures, dataset loaders, concrete callbacks/loggers, and the Typer CLI entry points, facilitating extensibility.
