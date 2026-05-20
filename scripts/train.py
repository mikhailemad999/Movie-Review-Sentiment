"""
scripts/train.py — Training entry-point with Typer + Rich output.

Usage:
    python scripts/train.py --config configs/default.yaml
    python scripts/train.py --config configs/default.yaml --epochs 20 --lr 5e-4
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sentiment.config import AppConfig
from sentiment.training.trainer import Trainer

app     = typer.Typer(help="GRU Sentiment — Training CLI", add_completion=False)
console = Console()


@app.command()
def train(
    config: Path         = typer.Option(Path("configs/default.yaml"), help="YAML config path"),
    epochs: Optional[int]   = typer.Option(None, help="Override training epochs"),
    lr:     Optional[float] = typer.Option(None, help="Override learning rate"),
    batch:  Optional[int]   = typer.Option(None, help="Override batch size"),
) -> None:
    """Train the GRU sentiment model from scratch."""
    console.rule("[bold cyan]GRU Sentiment — Training")
    cfg = AppConfig.from_yaml(config)

    if epochs: cfg.training.epochs          = epochs
    if lr:     cfg.training.learning_rate   = lr
    if batch:  cfg.training.batch_size      = batch

    table = Table(title="Active Configuration", header_style="bold magenta")
    table.add_column("Parameter", style="cyan")
    table.add_column("Value",     style="green")
    for k, v in [
        ("Model", cfg.model.name), ("Epochs", cfg.training.epochs),
        ("Batch Size", cfg.training.batch_size), ("LR", cfg.training.learning_rate),
        ("GRU Units", cfg.model.gru_units), ("Attention", cfg.model.attention),
        ("MLflow URI", cfg.logging.mlflow_uri),
    ]:
        table.add_row(str(k), str(v))
    console.print(table)

    with console.status("[bold green]Training in progress ..."):
        result = Trainer(cfg).run()

    console.print(Panel(
        f"[bold green]Training complete![/]\n\n"
        f"  Run ID       : [cyan]{result.run_id}[/]\n"
        f"  Val Accuracy : [yellow]{result.best_val_accuracy * 100:.2f}%[/]\n"
        f"  Val AUC      : [yellow]{result.best_val_auc:.4f}[/]\n"
        f"  Model        : [blue]{result.model_path}[/]",
        title="Results", border_style="green",
    ))


if __name__ == "__main__":
    app()
