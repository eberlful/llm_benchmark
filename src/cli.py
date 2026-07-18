import os
import time
import yaml
from pathlib import Path
from typing import Optional
import torch
import typer
import tiktoken
from rich.console import Console

from src.models.gpt import GPTModel, GPTConfig
from src.models.pam import PAMConfig, PAMModel
from src.datasets.shakespeare import ShakespeareDataset
from src.trainer import Trainer
from src.callbacks.terminal_logger import TerminalLogger
from src.callbacks.tensorboard_logger import TensorBoardLogger
from src.callbacks.checkpoint import CheckpointCallback

app = typer.Typer(help="🔥 NanoGPT Benchmark CLI Interface 🔥")
console = Console()

@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to the YAML configuration file.", exists=True, file_okay=True, dir_okay=False, readable=True),
    learning_rate: Optional[float] = typer.Option(None, "--learning-rate", "-lr", help="Override optimizer learning rate."),
    steps: Optional[int] = typer.Option(None, "--steps", "-s", help="Override maximum iterations/steps."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Override training batch size."),
):
    """
    🚀 Start training a model with a YAML config and optional parameter overrides.
    """
    console.print("[bold green]📥 Loading configuration...[/bold green]")
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to parse config file: {e}[/bold red]")
        raise typer.Exit(code=1)

    # 1. Dataset Config
    dataset_cfg = config.get("dataset", {})
    data_dir = dataset_cfg.get("data_dir", "data/shakespeare")
    console.print(f"📊 Dataset directory: [cyan]{data_dir}[/cyan]")
    dataset = ShakespeareDataset(data_dir=data_dir)

    # 2. Model Config
    model_cfg = config.get("model", {}).copy()
    if "dropout" in model_cfg:
        model_cfg["dropout"] = float(model_cfg["dropout"])
    
    model_type = model_cfg.pop("type", "gpt")
    if model_type == "pam":
        if "block_size" in model_cfg:
            model_cfg["max_seq_len"] = model_cfg["block_size"]
        if "n_embd" in model_cfg:
            model_cfg["dim"] = model_cfg["n_embd"]
        pam_config = PAMConfig(**model_cfg)
        console.print(f"🤖 Initializing PAM model architecture (n_layer={pam_config.n_layer}, n_head={pam_config.n_head}, dim={pam_config.dim})...")
        model = PAMModel(pam_config)
    else:
        gpt_config = GPTConfig(**model_cfg)
        console.print(f"🤖 Initializing GPT model architecture (n_layer={gpt_config.n_layer}, n_head={gpt_config.n_head}, n_embd={gpt_config.n_embd})...")
        model = GPTModel(gpt_config)

    # 3. Trainer & Optimizer configs
    trainer_cfg = config.get("trainer", {}).copy()
    
    # Extract optimizer params
    opt_cfg = config.get("optimizer", {})
    weight_decay = float(opt_cfg.get("weight_decay", 0.01))
    lr = float(learning_rate if learning_rate is not None else opt_cfg.get("learning_rate", 6e-4))
    betas = tuple(opt_cfg.get("betas", [0.9, 0.95]))
    
    # Cast trainer float configs if present
    for k in ["learning_rate", "min_lr", "grad_clip"]:
        if k in trainer_cfg:
            trainer_cfg[k] = float(trainer_cfg[k])

    device = trainer_cfg.pop("device", "cpu")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device_type = "cuda" if "cuda" in device else "cpu"

    console.print(f"⚡ Device set to [yellow]{device}[/yellow], configuring optimizer...")
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=betas,
        device_type=device_type
    )

    # Apply overrides to trainer config
    if steps is not None:
        trainer_cfg["max_iters"] = steps
    if batch_size is not None:
        trainer_cfg["batch_size"] = batch_size

    # Create a unique output directory under runs/
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("runs", f"run_{timestamp}")
    console.print(f"📂 Output run directory: [cyan]{out_dir}[/cyan]")

    # Setup callbacks
    log_interval = trainer_cfg.pop("log_interval", 10)
    callbacks = [
        TerminalLogger(log_interval=log_interval),
        TensorBoardLogger(log_interval=log_interval),
        CheckpointCallback()
    ]

    # Pop explicit params and construct Trainer
    trainer_cfg.pop("out_dir", None)
    trainer_cfg.pop("learning_rate", None)
    
    max_iters = trainer_cfg.pop("max_iters", 2000)
    batch_size_val = trainer_cfg.pop("batch_size", 12)
    block_size = trainer_cfg.pop("block_size", 1024)

    trainer = Trainer(
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        max_iters=max_iters,
        batch_size=batch_size_val,
        block_size=block_size,
        learning_rate=lr,
        out_dir=out_dir,
        callbacks=callbacks,
        device=device,
        **trainer_cfg
    )

    console.print("[bold green]🏁 Starting training loop...[/bold green]")
    trainer.train()
    console.print("[bold green]🏆 Training finished successfully![/bold green]")

@app.command()
def eval(
    checkpoint_path: Path = typer.Option(..., "--checkpoint-path", "-c", help="Path to the saved checkpoint.", exists=True, file_okay=True, dir_okay=False, readable=True),
    data_dir: Path = typer.Option(Path("data/shakespeare"), "--data-dir", "-d", help="Directory of dataset files.", exists=True, file_okay=False, dir_okay=True, readable=True),
    batch_size: int = typer.Option(12, "--batch-size", "-b", help="Batch size for evaluation."),
    eval_iters: int = typer.Option(200, "--eval-iters", "-e", help="Number of evaluation iterations to average."),
    device: str = typer.Option("auto", "--device", "-dev", help="Execution device (e.g. cpu, cuda)."),
):
    """
    🔍 Evaluate a model checkpoint on the validation split.
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    console.print(f"[bold green]🚀 Loading checkpoint from {checkpoint_path}...[/bold green]")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load checkpoint: {e}[/bold red]")
        raise typer.Exit(code=1)

    config = checkpoint.get("config")
    if config is None:
        console.print("[bold red]❌ Checkpoint does not contain config metadata.[/bold red]")
        raise typer.Exit(code=1)

    if isinstance(config, PAMConfig):
        console.print("🤖 Instantiating PAM model...")
        model = PAMModel(config)
    else:
        console.print("🤖 Instantiating GPT model...")
        model = GPTModel(config)
    
    state_dict = checkpoint["model"]
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    console.print(f"📊 Instantiating ShakespeareDataset from {data_dir}...")
    dataset = ShakespeareDataset(data_dir=str(data_dir))

    console.print("🔄 Running evaluation...")
    losses = torch.zeros(eval_iters)
    with torch.no_grad():
        for k in range(eval_iters):
            x, y = dataset.get_batch("val", batch_size, config.block_size, device)
            _, loss = model(x, y)
            losses[k] = loss.item()
            
    val_loss = losses.mean().item()
    console.print(f"🏆 [bold green]Average Validation Loss:[/bold green] [bold cyan]{val_loss:.4f}[/bold cyan]")

@app.command()
def inference(
    checkpoint_path: Path = typer.Option(..., "--checkpoint-path", "-c", help="Path to the saved checkpoint.", exists=True, file_okay=True, dir_okay=False, readable=True),
    prompt: Optional[str] = typer.Option(None, "--prompt", "-p", help="Text prompt to complete."),
    prompt_file: Optional[Path] = typer.Option(None, "--prompt-file", "-f", help="Path to file containing prompt text.", exists=True, file_okay=True, dir_okay=False, readable=True),
    max_new_tokens: int = typer.Option(200, "--max-new-tokens", "-n", help="Number of tokens to generate."),
    temperature: float = typer.Option(1.0, "--temperature", "-t", help="Sampling temperature."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", help="Keep only top k tokens."),
    device: str = typer.Option("auto", "--device", "-dev", help="Execution device (e.g. cpu, cuda)."),
):
    """
    ✨ Generate text starting from a prompt using a model checkpoint.
    """
    if prompt is not None and prompt_file is not None:
        console.print("[bold red]❌ Cannot specify both prompt string and prompt file.[/bold red]")
        raise typer.Exit(code=1)

    prompt_text = "\n"
    if prompt is not None:
        prompt_text = prompt
    elif prompt_file is not None:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                prompt_text = f.read()
        except Exception as e:
            console.print(f"[bold red]❌ Failed to read prompt file: {e}[/bold red]")
            raise typer.Exit(code=1)

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    console.print(f"[bold green]🚀 Loading checkpoint from {checkpoint_path}...[/bold green]")
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load checkpoint: {e}[/bold red]")
        raise typer.Exit(code=1)

    config = checkpoint.get("config")
    if config is None:
        console.print("[bold red]❌ Checkpoint does not contain config metadata.[/bold red]")
        raise typer.Exit(code=1)

    if isinstance(config, PAMConfig):
        console.print("🤖 Instantiating PAM model...")
        model = PAMModel(config)
    else:
        console.print("🤖 Instantiating GPT model...")
        model = GPTModel(config)
    
    state_dict = checkpoint["model"]
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    console.print("🔤 Encoding prompt using tiktoken...")
    enc = tiktoken.get_encoding("gpt2")
    start_ids = enc.encode(prompt_text, allowed_special={""} | set())
    x = torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...]

    console.print("🔮 Generating completions...")
    with torch.no_grad():
        y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
        
    generated_text = enc.decode(y[0].tolist())
    console.print("\n[bold yellow]✨ Generated Text Output: ✨[/bold yellow]")
    console.print(generated_text)

if __name__ == "__main__":
    app()
