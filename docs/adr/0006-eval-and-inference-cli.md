# 0006. Eval and Inference CLI

We decided that the `eval` and `inference` commands will load a model from a specified checkpoint path or output directory. The `eval` command will execute evaluation on the validation split. The `inference` command will accept parameter options (`--max-new-tokens`, `--temperature`, `--top-k`) and a prompt string or file path, generating output tokens from the loaded model.
