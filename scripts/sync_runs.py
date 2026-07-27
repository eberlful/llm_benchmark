#!/usr/bin/env python3
"""
Sync training runs with Google Drive for Google Colab / cloud training workflows.
Supports uploading completed/in-progress runs, downloading runs to continue training,
listing run status, and auto-mounting Google Drive in Google Colab.
"""

import os
import sys
import re
import shutil
import time
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="☁️ Sync training runs with Google Drive for Colab / cloud workflows")
console = Console()


def is_colab() -> bool:
    """Check if code is running inside Google Colab."""
    try:
        import google.colab  # type: ignore # noqa: F401
        return True
    except ImportError:
        return False


def mount_google_drive(mount_point: str = "/content/drive", force: bool = False) -> Optional[Path]:
    """
    Mount Google Drive in Google Colab environment if available.
    Returns path to drive root (e.g. /content/drive/MyDrive) or None if not in Colab.
    """
    if is_colab():
        try:
            from google.colab import drive  # type: ignore
            drive_root = Path(mount_point) / "MyDrive"
            if not drive_root.exists() or force:
                console.print(f"📌 Mounting Google Drive at [cyan]{mount_point}[/cyan]...")
                drive.mount(mount_point)
            if drive_root.exists():
                return drive_root
            return Path(mount_point)
        except Exception as e:
            console.print(f"[bold yellow]⚠️ Failed to mount Google Drive automatically: {e}[/bold yellow]")
            return None
    
    # Non-colab fallback checks
    env_drive = os.getenv("GOOGLE_DRIVE_DIR")
    if env_drive:
        return Path(env_drive).expanduser().resolve()
    
    colab_path = Path(mount_point) / "MyDrive"
    if colab_path.exists():
        return colab_path
    
    return None


def resolve_drive_dir(user_drive_dir: Optional[Path] = None, mount: bool = True) -> Path:
    """
    Resolve Google Drive runs directory path based on user option, env var, or Colab auto-mount.
    """
    if user_drive_dir is not None:
        return user_drive_dir.expanduser().resolve()
    
    env_dir = os.getenv("GOOGLE_DRIVE_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    
    if mount:
        drive_root = mount_google_drive()
        if drive_root is not None:
            return drive_root / "llm_benchmark_runs"
    
    # Fallback to local drive_runs directory if not in Colab and no env variable set
    return Path("/content/drive/MyDrive/llm_benchmark_runs") if is_colab() else Path("drive_runs").resolve()


def get_run_info(run_dir: Path) -> Dict[str, Any]:
    """
    Extract metadata from a run directory.
    """
    info: Dict[str, Any] = {
        "name": run_dir.name,
        "path": run_dir,
        "exists": run_dir.is_dir(),
        "mtime": 0.0,
        "formatted_mtime": "N/A",
        "checkpoints": [],
        "best_val_loss": None,
        "latest_step": None,
        "size_bytes": 0,
        "size_mb": 0.0,
    }
    
    if not run_dir.is_dir():
        return info
    
    info["mtime"] = run_dir.stat().st_mtime
    info["formatted_mtime"] = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(info["mtime"]))
    
    total_size = 0
    checkpoints = []
    best_loss = float("inf")
    max_step = -1
    
    for root, _, files in os.walk(run_dir):
        for f in files:
            fp = Path(root) / f
            try:
                size = fp.stat().st_size
                total_size += size
            except OSError:
                pass
            
            if f.endswith(".pt"):
                checkpoints.append(f)
                # Parse step and val loss from filename like ckpt_step_100_val_loss_1.2345.pt
                step_match = re.search(r"step_(\d+)", f)
                if step_match:
                    step_val = int(step_match.group(1))
                    if step_val > max_step:
                        max_step = step_val
                
                loss_match = re.search(r"val_loss_([0-9]+(?:\.[0-9]+)?)", f)
                if loss_match:
                    try:
                        loss_val = float(loss_match.group(1))
                        if loss_val < best_loss:
                            best_loss = loss_val
                    except ValueError:
                        pass

    info["size_bytes"] = total_size
    info["size_mb"] = total_size / (1024 * 1024)
    info["checkpoints"] = checkpoints
    info["latest_step"] = max_step if max_step >= 0 else None
    info["best_val_loss"] = best_loss if best_loss != float("inf") else None
    return info


def list_run_dirs(base_dir: Path) -> List[Path]:
    """List all run directories under base_dir sorted by modified time (newest first)."""
    if not base_dir.exists() or not base_dir.is_dir():
        return []
    
    dirs = [d for d in base_dir.iterdir() if d.is_dir()]
    dirs.sort(key=lambda d: d.stat().st_mtime, reverse=True)
    return dirs


def find_latest_run(base_dir: Path) -> Optional[Path]:
    """Find the most recently modified run directory in base_dir."""
    dirs = list_run_dirs(base_dir)
    return dirs[0] if dirs else None


def resolve_run_path(run_name_or_path: Optional[str], base_dir: Path, latest: bool = False) -> Tuple[Optional[Path], str]:
    """
    Resolve run path from name, path, or latest flag.
    Returns (resolved_path, run_name).
    """
    if latest or run_name_or_path is None:
        latest_dir = find_latest_run(base_dir)
        if latest_dir is None:
            return None, ""
        return latest_dir, latest_dir.name
    
    # Direct path or relative to base_dir
    candidate = Path(run_name_or_path)
    if candidate.is_dir():
        return candidate, candidate.name
    
    target = base_dir / run_name_or_path
    if target.is_dir():
        return target, target.name
    
    return None, run_name_or_path


def copy_sync_directory(src_dir: Path, dst_dir: Path) -> None:
    """Copy all files and directories from src_dir to dst_dir, overwriting existing files."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    for root, dirs, files in os.walk(src_dir):
        rel_path = Path(root).relative_to(src_dir)
        target_dir = dst_dir / rel_path
        target_dir.mkdir(parents=True, exist_ok=True)
        
        for file in files:
            src_file = Path(root) / file
            dst_file = target_dir / file
            # Copy file if it doesn't exist or size/mtime differs
            if not dst_file.exists() or src_file.stat().st_size != dst_file.stat().st_size or src_file.stat().st_mtime > dst_file.stat().st_mtime:
                shutil.copy2(src_file, dst_file)


# -----------------------------------------------------------------------------
# CLI Commands
# -----------------------------------------------------------------------------

@app.command("upload")
def upload_cmd(
    run_name: Optional[str] = typer.Argument(
        None,
        help="Name or path of the local run to upload (e.g. run_20260727_103000). If omitted, uploads latest run unless --all is set."
    ),
    drive_dir: Optional[Path] = typer.Option(
        None, "--drive-dir", "-d",
        help="Path to Google Drive runs directory. Defaults to GOOGLE_DRIVE_DIR or /content/drive/MyDrive/llm_benchmark_runs."
    ),
    local_dir: Path = typer.Option(
        Path("runs"), "--local-dir", "-l",
        help="Path to local runs directory."
    ),
    latest: bool = typer.Option(
        False, "--latest",
        help="Upload the latest local run directory."
    ),
    all_runs: bool = typer.Option(
        False, "--all", "-a",
        help="Upload all local run directories."
    ),
    mount: bool = typer.Option(
        True, "--mount/--no-mount",
        help="Attempt auto-mounting Google Drive in Google Colab."
    ),
):
    """
    📤 Upload local training run(s) to Google Drive.
    """
    target_drive_dir = resolve_drive_dir(drive_dir, mount=mount)
    local_dir = local_dir.resolve()
    
    if not local_dir.exists():
        console.print(f"[bold red]❌ Local runs directory '{local_dir}' does not exist.[/bold red]")
        raise typer.Exit(code=1)
    
    runs_to_upload: List[Tuple[Path, str]] = []
    
    if all_runs:
        dirs = list_run_dirs(local_dir)
        if not dirs:
            console.print(f"[yellow]⚠️ No run directories found in '{local_dir}'.[/yellow]")
            return
        runs_to_upload = [(d, d.name) for d in dirs]
    else:
        run_path, name = resolve_run_path(run_name, local_dir, latest=latest)
        if run_path is None or not run_path.exists():
            console.print(f"[bold red]❌ Could not find local run '{run_name or 'latest'}' in '{local_dir}'.[/bold red]")
            raise typer.Exit(code=1)
        runs_to_upload = [(run_path, name)]
    
    console.print(f"🚀 Uploading [bold cyan]{len(runs_to_upload)}[/bold cyan] run(s) to Drive: [yellow]{target_drive_dir}[/yellow]")
    
    for src_path, name in runs_to_upload:
        dst_path = target_drive_dir / name
        console.print(f"  📦 Syncing [cyan]{name}[/cyan] -> [yellow]{dst_path}[/yellow]...")
        copy_sync_directory(src_path, dst_path)
        console.print(f"  [bold green]✅ Uploaded {name}[/bold green]")
    
    console.print("[bold green]🏆 All requested runs uploaded successfully![/bold green]")


@app.command("download")
def download_cmd(
    run_name: Optional[str] = typer.Argument(
        None,
        help="Name or path of the run to download from Google Drive. If omitted, downloads latest run unless --all is set."
    ),
    drive_dir: Optional[Path] = typer.Option(
        None, "--drive-dir", "-d",
        help="Path to Google Drive runs directory."
    ),
    local_dir: Path = typer.Option(
        Path("runs"), "--local-dir", "-l",
        help="Path to local runs directory."
    ),
    latest: bool = typer.Option(
        False, "--latest",
        help="Download the latest run from Google Drive."
    ),
    all_runs: bool = typer.Option(
        False, "--all", "-a",
        help="Download all runs from Google Drive."
    ),
    mount: bool = typer.Option(
        True, "--mount/--no-mount",
        help="Attempt auto-mounting Google Drive in Google Colab."
    ),
):
    """
    📥 Download training run(s) from Google Drive to local storage.
    """
    target_drive_dir = resolve_drive_dir(drive_dir, mount=mount)
    local_dir = local_dir.resolve()
    
    if not target_drive_dir.exists():
        console.print(f"[bold red]❌ Google Drive runs directory '{target_drive_dir}' does not exist.[/bold red]")
        raise typer.Exit(code=1)
    
    runs_to_download: List[Tuple[Path, str]] = []
    
    if all_runs:
        dirs = list_run_dirs(target_drive_dir)
        if not dirs:
            console.print(f"[yellow]⚠️ No run directories found in Drive folder '{target_drive_dir}'.[/yellow]")
            return
        runs_to_download = [(d, d.name) for d in dirs]
    else:
        run_path, name = resolve_run_path(run_name, target_drive_dir, latest=latest)
        if run_path is None or not run_path.exists():
            console.print(f"[bold red]❌ Could not find run '{run_name or 'latest'}' in Drive folder '{target_drive_dir}'.[/bold red]")
            raise typer.Exit(code=1)
        runs_to_download = [(run_path, name)]
    
    console.print(f"📥 Downloading [bold cyan]{len(runs_to_download)}[/bold cyan] run(s) from Drive to local [yellow]{local_dir}[/yellow]")
    
    for src_path, name in runs_to_download:
        dst_path = local_dir / name
        console.print(f"  📦 Syncing [yellow]{src_path}[/yellow] -> [cyan]{dst_path}[/cyan]...")
        copy_sync_directory(src_path, dst_path)
        console.print(f"  [bold green]✅ Downloaded {name}[/bold green]")
    
    console.print("[bold green]🏆 All requested runs downloaded successfully![/bold green]")


@app.command("list")
def list_cmd(
    drive_dir: Optional[Path] = typer.Option(
        None, "--drive-dir", "-d",
        help="Path to Google Drive runs directory."
    ),
    local_dir: Path = typer.Option(
        Path("runs"), "--local-dir", "-l",
        help="Path to local runs directory."
    ),
    mount: bool = typer.Option(
        True, "--mount/--no-mount",
        help="Attempt auto-mounting Google Drive in Google Colab."
    ),
):
    """
    📊 List local and Google Drive training runs with checkpoint details.
    """
    target_drive_dir = resolve_drive_dir(drive_dir, mount=mount)
    local_dir = local_dir.resolve()
    
    local_run_dirs = {d.name: d for d in list_run_dirs(local_dir)}
    drive_run_dirs = {d.name: d for d in list_run_dirs(target_drive_dir)} if target_drive_dir.exists() else {}
    
    all_run_names = sorted(set(local_run_dirs.keys()) | set(drive_run_dirs.keys()), reverse=True)
    
    if not all_run_names:
        console.print("[yellow]⚠️ No training runs found locally or in Google Drive.[/yellow]")
        return
    
    table = Table(title="📊 Training Runs Status", show_header=True, header_style="bold magenta")
    table.add_column("Run Name", style="cyan", no_wrap=True)
    table.add_column("Local", justify="center")
    table.add_column("Drive", justify="center")
    table.add_column("Latest Step", justify="right")
    table.add_column("Best Val Loss", justify="right")
    table.add_column("Size (MB)", justify="right")
    table.add_column("Last Modified", style="dim")
    
    for name in all_run_names:
        loc_dir = local_run_dirs.get(name)
        drv_dir = drive_run_dirs.get(name)
        
        info = get_run_info(loc_dir or drv_dir)  # type: ignore
        
        loc_status = "✅" if loc_dir else "❌"
        drv_status = "✅" if drv_dir else "❌"
        
        step_str = str(info["latest_step"]) if info["latest_step"] is not None else "-"
        loss_str = f"{info['best_val_loss']:.4f}" if info["best_val_loss"] is not None else "-"
        size_str = f"{info['size_mb']:.1f}"
        
        table.add_row(
            name,
            loc_status,
            drv_status,
            step_str,
            loss_str,
            size_str,
            info["formatted_mtime"]
        )
    
    console.print(table)
    console.print(f"📂 Local runs directory: [cyan]{local_dir}[/cyan]")
    console.print(f"☁️ Drive runs directory: [yellow]{target_drive_dir}[/yellow]")


@app.command("mount")
def mount_cmd(
    mount_point: str = typer.Option(
        "/content/drive", "--mount-point", "-m",
        help="Mount point for Google Drive."
    )
):
    """
    🔗 Mount Google Drive in Google Colab.
    """
    drive_root = mount_google_drive(mount_point=mount_point, force=True)
    if drive_root:
        console.print(f"[bold green]✅ Google Drive mounted successfully at '{drive_root}'[/bold green]")
    else:
        console.print("[bold red]❌ Failed to mount Google Drive. Are you running inside Google Colab?[/bold red]")


if __name__ == "__main__":
    app()
