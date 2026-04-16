from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
from torch import nn

from src.data import build_har_activity_dataloaders
from src.models import CNN1DClassifier, CNNBiLSTMClassifier, CNNBiLSTMAttentionClassifier
from src.training.utils import compute_classification_metrics, ensure_dir, save_history_csv, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HAR activity multiclass classifier")
    parser.add_argument("--config", type=str, default="configs/har_activity_phase2.yaml")
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


def build_model(config: Dict[str, Any], num_classes: int, in_channels: int, seq_len: int) -> nn.Module:
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "cnn_bilstm_attention")).lower()

    if model_name == "cnn1d":
        return CNN1DClassifier(
            in_channels=int(model_cfg.get("in_channels", in_channels)),
            seq_len=int(model_cfg.get("seq_len", seq_len)),
            hidden_dim=int(model_cfg.get("hidden_dim", 128)),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if model_name == "cnn_bilstm":
        return CNNBiLSTMClassifier(
            in_channels=int(model_cfg.get("in_channels", in_channels)),
            conv_channels=int(model_cfg.get("conv_channels", 128)),
            lstm_hidden_dim=int(model_cfg.get("lstm_hidden_dim", 128)),
            lstm_layers=int(model_cfg.get("lstm_layers", 1)),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.3)),
        )

    if model_name == "cnn_bilstm_attention":
        return CNNBiLSTMAttentionClassifier(
            in_channels=int(model_cfg.get("in_channels", in_channels)),
            conv_channels=int(model_cfg.get("conv_channels", 128)),
            lstm_hidden_dim=int(model_cfg.get("lstm_hidden_dim", 128)),
            lstm_layers=int(model_cfg.get("lstm_layers", 1)),
            num_classes=num_classes,
            dropout=float(model_cfg.get("dropout", 0.3)),
            attention_heads=int(model_cfg.get("attention_heads", 4)),
        )

    raise ValueError(f"Unsupported model.name: {model_name}")


def class_weights(y: np.ndarray, num_classes: int) -> torch.Tensor:
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts = np.maximum(counts, 1.0)
    weights = counts.sum() / (len(counts) * counts)
    weights = weights / np.mean(weights)
    return torch.tensor(weights, dtype=torch.float32)


class WeightedFocalCrossEntropy(nn.Module):
    def __init__(self, class_weight: Optional[torch.Tensor] = None, gamma: float = 2.0) -> None:
        super().__init__()
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(weight=class_weight, reduction="none")

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = self.ce(logits, targets)
        probs = torch.softmax(logits, dim=1)
        pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        focal = (1.0 - pt).pow(self.gamma)
        return (focal * ce).mean()


def run_epoch(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: Optional[torch.optim.Optimizer] = None,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
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
        bs = int(y.size(0))

        running_loss += float(loss.item()) * bs
        total_samples += bs
        all_targets.append(y.detach().cpu().numpy())
        all_preds.append(preds.detach().cpu().numpy())

    if total_samples == 0:
        raise RuntimeError("No samples processed in epoch")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_preds)

    metrics = compute_classification_metrics(y_true, y_pred)
    metrics["loss"] = running_loss / total_samples
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


def collect_detailed_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> Dict[str, Any]:
    p, r, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=list(range(len(class_names))),
        zero_division=0,
    )
    per_class = {
        class_names[i]: {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i in range(len(class_names))
    }
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names)))).tolist()
    return {"per_class": per_class, "confusion_matrix": cm}


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(args.device)

    loaders = build_har_activity_dataloaders(config)
    input_shape = loaders["input_shape"]
    class_names = list(loaders["label_encoding"].classes)
    print(f"[run] input_shape={input_shape} metadata={loaders['metadata']}")

    model = build_model(
        config=config,
        num_classes=int(loaders["num_classes"]),
        in_channels=int(input_shape["channels"]),
        seq_len=int(input_shape["seq_len"]),
    )
    print(f"[run] model={config.get('model', {}).get('name', 'cnn_bilstm_attention')} classes={len(class_names)} device={device}")
    model.to(device)

    train_cfg = config.get("train", {})
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 6))

    loss_cfg = config.get("loss", {})
    use_weights = bool(loss_cfg.get("class_weighting", True))
    balanced_sampling = bool(config.get("data", {}).get("balanced_sampling", False))
    disable_weights_when_balanced = bool(loss_cfg.get("disable_class_weight_when_balanced_sampling", True))
    if balanced_sampling and use_weights and disable_weights_when_balanced:
        print("[loss] balanced_sampling=true, disabling class_weighting to avoid over-correction")
        use_weights = False
    loss_name = str(loss_cfg.get("name", "cross_entropy")).lower()
    focal_gamma = float(loss_cfg.get("focal_gamma", 2.5))

    cls_w = None
    if use_weights:
        cls_w = class_weights(loaders["targets"]["train"], num_classes=int(loaders["num_classes"])).to(device)

    if loss_name == "focal":
        criterion = WeightedFocalCrossEntropy(class_weight=cls_w, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_w)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_cfg = config.get("logging", {})
    output_dir = ensure_dir(Path(log_cfg.get("output_dir", "outputs")) / log_cfg.get("run_name", "har_activity_phase2"))

    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    wait = 0
    history = []

    for epoch in range(1, epochs + 1):
        train_m = run_epoch(
            model=model,
            loader=loaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=args.max_batches,
        )

        with torch.no_grad():
            val_m = run_epoch(
                model=model,
                loader=loaders["val"],
                criterion=criterion,
                device=device,
                optimizer=None,
                max_batches=args.max_batches,
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_m["loss"],
                "train_acc": train_m["accuracy"],
                "train_macro_f1": train_m["macro_f1"],
                "val_loss": val_m["loss"],
                "val_acc": val_m["accuracy"],
                "val_macro_f1": val_m["macro_f1"],
            }
        )

        score = float(val_m["macro_f1"])
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1

        print(
            f"Epoch {epoch:03d} | train_f1={train_m['macro_f1']:.4f} train_acc={train_m['accuracy']:.4f} "
            f"| val_f1={val_m['macro_f1']:.4f} val_acc={val_m['accuracy']:.4f}"
        )

        if wait >= patience:
            print(f"Early stopping at epoch {epoch}; best_epoch={best_epoch}")
            break

    if best_state is None:
        raise RuntimeError("No best checkpoint captured")

    model.load_state_dict(best_state)
    with torch.no_grad():
        test_m = run_epoch(
            model=model,
            loader=loaders["test"],
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=args.max_batches,
        )

    detailed_test = collect_detailed_metrics(test_m["y_true"], test_m["y_pred"], class_names=class_names)

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_shape": loaders["input_shape"],
            "num_classes": int(loaders["num_classes"]),
            "label_encoding": {
                "classes": class_names,
                "to_index": loaders["label_encoding"].to_index,
            },
        },
        output_dir / "best_model.pt",
    )

    save_history_csv(output_dir / "history.csv", history)
    save_json(
        output_dir / "metrics.json",
        {
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1,
            "test": {
                "accuracy": test_m["accuracy"],
                "macro_f1": test_m["macro_f1"],
                "loss": test_m["loss"],
            },
            "detailed_test": detailed_test,
            "input_shape": loaders["input_shape"],
            "label_encoding": {
                "classes": class_names,
                "to_index": loaders["label_encoding"].to_index,
            },
            "dataset": loaders["metadata"],
            "note": "Phase 2 multiclass activity model with gamma-2.5 attention baseline stack.",
        },
    )
    save_json(output_dir / "normalization.json", loaders["normalization"])

    print("Saved artifacts:")
    print(f" - {output_dir / 'best_model.pt'}")
    print(f" - {output_dir / 'history.csv'}")
    print(f" - {output_dir / 'metrics.json'}")
    print(f" - {output_dir / 'normalization.json'}")


if __name__ == "__main__":
    main()
