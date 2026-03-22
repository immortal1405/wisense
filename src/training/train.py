from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn

from src.data.csi_dataset import build_dataloaders
from src.models import CNN1DClassifier, CNNBiLSTMClassifier
from src.training.utils import (
    compute_classification_metrics,
    ensure_dir,
    save_history_csv,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Wi-Sense CSI model")
    parser.add_argument("--config", type=str, default="configs/base.yaml", help="Path to YAML config")
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs")
    parser.add_argument("--max-batches", type=int, default=None, help="Limit batches per epoch for quick smoke tests")
    parser.add_argument("--device", type=str, default=None, help="Force device: cpu or cuda")
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: Dict[str, Any], num_classes: int) -> nn.Module:
    model_cfg = config.get("model", {})
    name = model_cfg.get("name", "cnn1d")
    if name == "cnn1d":
        return CNN1DClassifier(
            in_channels=int(model_cfg.get("in_channels", 2)),
            seq_len=int(model_cfg.get("seq_len", 52)),
            hidden_dim=int(model_cfg.get("hidden_dim", 64)),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.2)),
        )

    if name == "cnn_bilstm":
        return CNNBiLSTMClassifier(
            in_channels=int(model_cfg.get("in_channels", 2)),
            conv_channels=int(model_cfg.get("conv_channels", 64)),
            lstm_hidden_dim=int(model_cfg.get("lstm_hidden_dim", 64)),
            lstm_layers=int(model_cfg.get("lstm_layers", 1)),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.2)),
        )

    raise ValueError(f"Unsupported model name: {name}")


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)

    all_targets = []
    all_preds = []
    running_loss = 0.0
    total_samples = 0

    for batch_idx, (x, y) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        logits = model(x)
        loss = criterion(logits, y)

        if is_train:
            loss.backward()
            optimizer.step()

        preds = torch.argmax(logits, dim=1)

        batch_size = y.size(0)
        running_loss += float(loss.item()) * batch_size
        total_samples += int(batch_size)
        all_targets.append(y.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

    if total_samples == 0:
        raise RuntimeError("No samples processed in epoch")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)
    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / total_samples
    return metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    seed_everything(seed)

    device = resolve_device(args.device)
    loaders = build_dataloaders(config)

    model = build_model(config, num_classes=int(loaders["num_classes"]))
    model.to(device)

    train_cfg = config.get("train", {})
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 8))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_cfg = config.get("logging", {})
    output_dir = ensure_dir(Path(log_cfg.get("output_dir", "outputs")) / log_cfg.get("run_name", "run"))

    history = []
    best_score = -1.0
    best_epoch = -1
    best_state = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model=model,
            loader=loaders["train_loader"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_batches=args.max_batches,
        )

        row = {
            "epoch": float(epoch),
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
        }

        val_loader = loaders.get("val_loader")
        if val_loader is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model=model,
                    loader=val_loader,
                    criterion=criterion,
                    optimizer=None,
                    device=device,
                    max_batches=args.max_batches,
                )
            row.update(
                {
                    "val_loss": val_metrics["loss"],
                    "val_acc": val_metrics["accuracy"],
                    "val_macro_f1": val_metrics["macro_f1"],
                }
            )
            score = float(val_metrics["macro_f1"])
        else:
            score = float(train_metrics["macro_f1"])

        history.append(row)

        if score > best_score:
            best_score = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        print(
            f"epoch={epoch} train_loss={row['train_loss']:.4f} train_f1={row['train_macro_f1']:.4f} "
            + (
                f"val_loss={row['val_loss']:.4f} val_f1={row['val_macro_f1']:.4f}"
                if "val_loss" in row
                else ""
            )
        )

        if patience_count >= patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch} best_score={best_score:.4f}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics = run_epoch(
            model=model,
            loader=loaders["test_loader"],
            criterion=criterion,
            optimizer=None,
            device=device,
            max_batches=args.max_batches,
        )

    torch.save(model.state_dict(), output_dir / "best_model.pt")
    save_history_csv(output_dir / "history.csv", history)

    summary = {
        "config_path": str(args.config),
        "device": str(device),
        "seed": seed,
        "split_sizes": loaders["split_sizes"],
        "labels": loaders["label_encoding"].classes,
        "best_epoch": best_epoch,
        "best_score": best_score,
        "test": test_metrics,
    }
    save_json(output_dir / "summary.json", summary)

    print("Training complete")
    print(f"best_epoch={best_epoch} best_score={best_score:.4f}")
    print(
        "test_loss={:.4f} test_acc={:.4f} test_macro_f1={:.4f}".format(
            test_metrics["loss"], test_metrics["accuracy"], test_metrics["macro_f1"]
        )
    )
    print(f"Artifacts written to: {output_dir}")


if __name__ == "__main__":
    main()
