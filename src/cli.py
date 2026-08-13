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
from src.models.xlstm import xLSTMConfig, xLSTMModel
from src.models.bdh import BDHConfig, BDHModel
from src.datasets.shakespeare import ShakespeareDataset
from src.trainer import Trainer
from src.callbacks.terminal_logger import TerminalLogger
from src.callbacks.tensorboard_logger import TensorBoardLogger
from src.callbacks.checkpoint import CheckpointCallback
from scripts.sync_runs import app as sync_app

app = typer.Typer(help="🔥 NanoGPT Benchmark CLI Interface 🔥")
app.add_typer(sync_app, name="sync", help="☁️ Sync training runs with Google Drive")
console = Console()


def instantiate_model(config) -> torch.nn.Module:
    """
    Instantiate model from config object (PAMConfig, xLSTMConfig, BDHConfig, or GPTConfig).
    """
    if isinstance(config, PAMConfig):
        console.print(f"🤖 Instantiating PAM model architecture (n_layer={config.n_layer}, n_head={config.n_head}, dim={config.dim})...")
        return PAMModel(config)
    elif isinstance(config, xLSTMConfig):
        console.print(f"🤖 Instantiating xLSTM model architecture (n_layer={config.n_layer}, num_heads={config.num_heads}, n_embd={config.n_embd}, pattern={config.block_type_pattern})...")
        return xLSTMModel(config)
    elif isinstance(config, BDHConfig):
        console.print(f"🤖 Instantiating BDH model architecture (n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd}, mlp_multiplier={config.mlp_internal_dim_multiplier})...")
        return BDHModel(config)
    else:
        console.print(f"🤖 Instantiating GPT model architecture (n_layer={config.n_layer}, n_head={config.n_head}, n_embd={config.n_embd})...")
        return GPTModel(config)


def create_model_from_dict(model_cfg: dict) -> torch.nn.Module:
    """
    Create a model and its configuration from a dictionary.
    """
    model_cfg = model_cfg.copy()
    if "dropout" in model_cfg:
        model_cfg["dropout"] = float(model_cfg["dropout"])

    model_type = model_cfg.pop("type", "gpt")
    if model_type == "pam":
        if "block_size" in model_cfg:
            model_cfg["max_seq_len"] = model_cfg["block_size"]
        if "n_embd" in model_cfg:
            model_cfg["dim"] = model_cfg["n_embd"]
        config = PAMConfig(**model_cfg)
    elif model_type == "xlstm":
        if "n_head" in model_cfg and "num_heads" not in model_cfg:
            model_cfg["num_heads"] = model_cfg.pop("n_head")
        config = xLSTMConfig(**model_cfg)
    elif model_type == "bdh":
        config = BDHConfig(**model_cfg)
    else:
        config = GPTConfig(**model_cfg)

    return instantiate_model(config)


@app.command()
def train(
    config_path: Path = typer.Argument(..., help="Path to the YAML configuration file.", exists=True, file_okay=True, dir_okay=False, readable=True),
    learning_rate: Optional[float] = typer.Option(None, "--learning-rate", "-lr", help="Override optimizer learning rate."),
    steps: Optional[int] = typer.Option(None, "--steps", "-s", help="Override maximum iterations/steps."),
    batch_size: Optional[int] = typer.Option(None, "--batch-size", "-b", help="Override training batch size."),
    eval_interval: Optional[int] = typer.Option(None, "--eval-interval", "-ei", help="Override evaluation interval."),
    eval_iters: Optional[int] = typer.Option(None, "--eval-iters", help="Override evaluation iterations."),
    log_interval: Optional[int] = typer.Option(None, "--log-interval", help="Override logging interval."),
    compile: Optional[bool] = typer.Option(None, "--compile/--no-compile", help="Compile model using PyTorch 2.0 (torch.compile)."),
    resume: Optional[Path] = typer.Option(None, "--resume", "-r", help="Path to checkpoint file or run directory to resume training from."),
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
    model_cfg = config.get("model", {})
    model = create_model_from_dict(model_cfg)

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

    model = model.to(device)

    console.print(f"⚡ Device set to [yellow]{device}[/yellow], configuring optimizer...")
    optimizer = model.configure_optimizers(
        weight_decay=weight_decay,
        learning_rate=lr,
        betas=betas,
        device_type=device_type
    )

    # Apply overrides to trainer config and config dict
    if learning_rate is not None:
        config.setdefault("optimizer", {})["learning_rate"] = lr
    if steps is not None:
        trainer_cfg["max_iters"] = steps
        config.setdefault("trainer", {})["max_iters"] = steps
    if batch_size is not None:
        trainer_cfg["batch_size"] = batch_size
        config.setdefault("trainer", {})["batch_size"] = batch_size
    if eval_interval is not None:
        trainer_cfg["eval_interval"] = eval_interval
        config.setdefault("trainer", {})["eval_interval"] = eval_interval
    if eval_iters is not None:
        trainer_cfg["eval_iters"] = eval_iters
        config.setdefault("trainer", {})["eval_iters"] = eval_iters
    if log_interval is not None:
        trainer_cfg["log_interval"] = log_interval
        config.setdefault("trainer", {})["log_interval"] = log_interval
    if compile is not None:
        trainer_cfg["compile"] = compile
        config.setdefault("trainer", {})["compile"] = compile

    # Create or reuse output directory under runs/
    if resume is not None:
        checkpoint_file = None
        if resume.is_dir():
            out_dir = str(resume)
            for fname in ["last_ckpt.pt", "best_ckpt.pt"]:
                if (resume / fname).is_file():
                    checkpoint_file = resume / fname
                    break
            if checkpoint_file is None:
                pts = sorted(resume.glob("ckpt_step_*.pt"))
                if pts:
                    checkpoint_file = pts[-1]
        elif resume.is_file():
            checkpoint_file = resume
            out_dir = str(resume.parent)

        if checkpoint_file is None or not checkpoint_file.exists():
            console.print(f"[bold red]❌ Could not find valid checkpoint to resume from at '{resume}'[/bold red]")
            raise typer.Exit(code=1)

        console.print(f"[bold green]🔄 Resuming training from checkpoint: [cyan]{checkpoint_file}[/cyan][/bold green]")
        ckpt = torch.load(checkpoint_file, map_location=device, weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        if "optimizer" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt.get("step", ckpt.get("steps", 0))
    else:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join("runs", f"run_{timestamp}")
        start_step = 0

    console.print(f"📂 Output run directory: [cyan]{out_dir}[/cyan]")
    os.makedirs(out_dir, exist_ok=True)
    config_save_path = os.path.join(out_dir, "config.yaml")
    with open(config_save_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    console.print(f"📄 Saved configuration to [cyan]{config_save_path}[/cyan]")

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
    compile_val = trainer_cfg.pop("compile", True)

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
        compile=compile_val,
        config=config,
        **trainer_cfg
    )
    if start_step > 0:
        trainer.run_state["iter_num"] = start_step


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

    model = instantiate_model(config)
    
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

    model = instantiate_model(config)
    
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
