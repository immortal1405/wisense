from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn

from src.data import build_activity_dataloaders
from src.models import APMLCNN1DClassifier
from src.training.utils import (
    compute_classification_metrics,
    ensure_dir,
    save_history_csv,
    save_json,
    seed_everything,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train APML activity model")
    parser.add_argument("--config", type=str, default="configs/activity_cnn_bilstm_apml.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def load_config(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_device(requested: Optional[str] = None) -> torch.device:
    if requested:
        return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_model(config: Dict[str, Any], num_classes: int, channels: int, seq_len: int) -> nn.Module:
    model_cfg = config.get("model", {})
    return APMLCNN1DClassifier(
        in_channels=int(model_cfg.get("in_channels", channels)),
        seq_len=int(model_cfg.get("seq_len", seq_len)),
        num_classes=num_classes,
    )


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


def _class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed = int(config.get("seed", 42))
    seed_everything(seed)

    device = resolve_device(args.device)
    loaders = build_activity_dataloaders(config)

    input_shape = loaders["input_shape"]
    model = build_model(
        config=config,
        num_classes=int(loaders["num_classes"]),
        channels=int(input_shape["channels"]),
        seq_len=int(input_shape["seq_len"]),
    )
    model.to(device)

    train_cfg = config.get("train", {})
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 8))

    loss_cfg = config.get("loss", {})
    use_class_weights = bool(loss_cfg.get("class_weighting", False))
    if use_class_weights:
        w = _class_weights(loaders["targets"]["train"], num_classes=int(loaders["num_classes"])).to(device)
        criterion = nn.CrossEntropyLoss(weight=w)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_cfg = config.get("logging", {})
    output_dir = ensure_dir(Path(log_cfg.get("output_dir", "outputs")) / log_cfg.get("run_name", "apml_notebook_run"))

    history = []
    best_score = -1.0
    best_epoch = -1
    best_state = None
    patience_count = 0

    for epoch in range(1, epochs + 1):
        train_metrics = run_epoch(
            model,
            loaders["train"],
            criterion,
            device,
            optimizer=optimizer,
            max_batches=args.max_batches,
        )

        if loaders["val"] is not None:
            with torch.no_grad():
                val_metrics = run_epoch(
                    model,
                    loaders["val"],
                    criterion,
                    device,
                    optimizer=None,
                    max_batches=args.max_batches,
                )
        else:
            val_metrics = train_metrics

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_acc": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_acc": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
        }
        history.append(row)

        val_score = float(val_metrics["macro_f1"])
        if val_score > best_score:
            best_score = val_score
            best_epoch = epoch
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={row['train_loss']:.4f} train_f1={row['train_macro_f1']:.4f} | "
            f"val_loss={row['val_loss']:.4f} val_f1={row['val_macro_f1']:.4f}"
        )

        if patience_count >= patience:
            print(f"Early stopping triggered at epoch {epoch} (patience={patience})")
            break

    if best_state is None:
        raise RuntimeError("Training did not produce a best checkpoint")

    model.load_state_dict(best_state)

    with torch.no_grad():
        test_metrics = run_epoch(
            model,
            loaders["test"],
            criterion,
            device,
            optimizer=None,
            max_batches=args.max_batches,
        )

    best_model_path = output_dir / "best_model.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "num_classes": int(loaders["num_classes"]),
            "input_shape": loaders["input_shape"],
            "label_encoding": loaders["label_encoding"],
        },
        best_model_path,
    )

    save_history_csv(output_dir / "history.csv", history)
    save_json(
        output_dir / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_score,
            "test": test_metrics,
            "label_encoding": {
                "classes": loaders["label_encoding"].classes,
                "to_index": loaders["label_encoding"].to_index,
            },
            "input_shape": loaders["input_shape"],
            "raw_slices": loaders["raw_slices"],
            "note": "Exact notebook-style APML slice pipeline.",
        },
    )
    save_json(output_dir / "normalization.json", loaders["normalization"])

    print("Saved:")
    print(f" - {best_model_path}")
    print(f" - {output_dir / 'history.csv'}")
    print(f" - {output_dir / 'metrics.json'}")
    print(f" - {output_dir / 'normalization.json'}")
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()
