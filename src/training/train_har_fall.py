from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn
from sklearn.metrics import precision_score, recall_score

from src.data import build_har_fall_dataloaders
from src.models import CNN1DClassifier, CNNBiLSTMClassifier, CNNBiLSTMAttentionClassifier
from src.training.utils import compute_classification_metrics, ensure_dir, save_history_csv, save_json, seed_everything


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HAR fall detector")
    parser.add_argument("--config", type=str, default="configs/har_fall_phase1.yaml")
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
    metrics["precision_pos"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["recall_pos"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))
    metrics["loss"] = running_loss / total_samples
    return metrics


def collect_probs(
    model: nn.Module,
    loader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_targets = []
    all_prob_pos = []

    with torch.no_grad():
        for batch_idx, (x, y) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            x = x.to(device)
            logits = model(x)
            prob_pos = torch.softmax(logits, dim=1)[:, 1]
            all_prob_pos.append(prob_pos.detach().cpu().numpy())
            all_targets.append(y.detach().cpu().numpy())

    y_true = np.concatenate(all_targets)
    p_pos = np.concatenate(all_prob_pos)
    return y_true, p_pos


def _fbeta(precision: float, recall: float, beta: float) -> float:
    b2 = beta * beta
    denom = b2 * precision + recall
    if denom <= 1e-12:
        return 0.0
    return float((1 + b2) * precision * recall / denom)


def tune_threshold(
    y_true: np.ndarray,
    p_pos: np.ndarray,
    objective: str = "macro_f1",
    beta: float = 2.0,
    min_recall: float = 0.0,
    grid_min: float = 0.05,
    grid_max: float = 0.95,
    grid_step: float = 0.01,
) -> Tuple[float, Dict[str, float]]:
    best_thr = 0.5
    best_metrics: Dict[str, float] = {"accuracy": 0.0, "macro_f1": -1.0, "precision_pos": 0.0, "recall_pos": 0.0}
    best_score = -1.0

    thresholds = np.arange(grid_min, grid_max + (0.5 * grid_step), grid_step)
    for thr in thresholds:
        y_pred = (p_pos >= thr).astype(np.int64)
        m = compute_classification_metrics(y_true, y_pred)
        m["precision_pos"] = float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))
        m["recall_pos"] = float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))

        if objective == "fbeta_pos":
            score = _fbeta(m["precision_pos"], m["recall_pos"], beta=beta)
        elif objective == "macro_f1_with_min_recall":
            score = m["macro_f1"] if m["recall_pos"] >= min_recall else -1.0
        else:
            score = m["macro_f1"]

        if (score > best_score) or (
            np.isclose(score, best_score) and m["macro_f1"] > best_metrics["macro_f1"]
        ):
            best_score = score
            best_metrics = m
            best_thr = float(thr)

    return best_thr, best_metrics


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    seed_everything(int(config.get("seed", 42)))
    device = resolve_device(args.device)

    loaders = build_har_fall_dataloaders(config)
    input_shape = loaders["input_shape"]
    print(f"[run] input_shape={input_shape} metadata={loaders['metadata']}")

    model = build_model(
        config=config,
        num_classes=int(loaders["num_classes"]),
        in_channels=int(input_shape["channels"]),
        seq_len=int(input_shape["seq_len"]),
    )
    print(f"[run] model={config.get('model', {}).get('name', 'cnn_bilstm')} device={device}")
    model.to(device)

    train_cfg = config.get("train", {})
    epochs = int(args.epochs if args.epochs is not None else train_cfg.get("epochs", 30))
    lr = float(train_cfg.get("lr", 1e-3))
    weight_decay = float(train_cfg.get("weight_decay", 1e-4))
    patience = int(train_cfg.get("early_stopping_patience", 6))

    loss_cfg = config.get("loss", {})
    use_weights = bool(loss_cfg.get("class_weighting", True))
    loss_name = str(loss_cfg.get("name", "cross_entropy")).lower()
    focal_gamma = float(loss_cfg.get("focal_gamma", 2.0))

    cls_w = None
    if use_weights:
        cls_w = class_weights(loaders["targets"]["train"], num_classes=int(loaders["num_classes"])).to(device)

    if loss_name == "focal":
        criterion = WeightedFocalCrossEntropy(class_weight=cls_w, gamma=focal_gamma)
    else:
        criterion = nn.CrossEntropyLoss(weight=cls_w)

    thr_cfg = config.get("threshold_tuning", {})
    thr_objective = str(thr_cfg.get("objective", "macro_f1")).lower()
    thr_beta = float(thr_cfg.get("beta", 2.0))
    thr_min_recall = float(thr_cfg.get("min_recall", 0.0))
    thr_grid_min = float(thr_cfg.get("grid_min", 0.05))
    thr_grid_max = float(thr_cfg.get("grid_max", 0.95))
    thr_grid_step = float(thr_cfg.get("grid_step", 0.01))

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    log_cfg = config.get("logging", {})
    output_dir = ensure_dir(Path(log_cfg.get("output_dir", "outputs")) / log_cfg.get("run_name", "har_fall_phase1"))

    best_f1 = -1.0
    best_epoch = -1
    best_state = None
    best_threshold = 0.5
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
        y_val_true, y_val_p = collect_probs(model=model, loader=loaders["val"], device=device, max_batches=args.max_batches)
        val_thr, val_thr_m = tune_threshold(
            y_val_true,
            y_val_p,
            objective=thr_objective,
            beta=thr_beta,
            min_recall=thr_min_recall,
            grid_min=thr_grid_min,
            grid_max=thr_grid_max,
            grid_step=thr_grid_step,
        )

        row = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_acc": train_m["accuracy"],
            "train_macro_f1": train_m["macro_f1"],
            "train_recall_fall": train_m["recall_pos"],
            "val_loss": val_m["loss"],
            "val_acc": val_m["accuracy"],
            "val_macro_f1": val_m["macro_f1"],
            "val_recall_fall": val_m["recall_pos"],
            "val_thr": val_thr,
            "val_macro_f1_thr": val_thr_m["macro_f1"],
            "val_recall_fall_thr": val_thr_m["recall_pos"],
        }
        history.append(row)

        score = float(val_thr_m["macro_f1"])
        if score > best_f1:
            best_f1 = score
            best_epoch = epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_threshold = val_thr
            wait = 0
        else:
            wait += 1

        print(
            f"Epoch {epoch:03d} | train_f1={train_m['macro_f1']:.4f} train_recall_fall={train_m['recall_pos']:.4f} "
            f"| val_f1={val_m['macro_f1']:.4f} val_f1_thr={val_thr_m['macro_f1']:.4f} "
            f"val_recall_fall_thr={val_thr_m['recall_pos']:.4f} thr={val_thr:.2f}"
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
    y_test_true, y_test_p = collect_probs(model=model, loader=loaders["test"], device=device, max_batches=args.max_batches)
    y_test_pred_thr = (y_test_p >= best_threshold).astype(np.int64)
    test_thr_m = compute_classification_metrics(y_test_true, y_test_pred_thr)
    test_thr_m["precision_pos"] = float(precision_score(y_test_true, y_test_pred_thr, pos_label=1, zero_division=0))
    test_thr_m["recall_pos"] = float(recall_score(y_test_true, y_test_pred_thr, pos_label=1, zero_division=0))

    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_shape": loaders["input_shape"],
            "num_classes": int(loaders["num_classes"]),
            "label_encoding": {
                "classes": loaders["label_encoding"].classes,
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
            "best_threshold": best_threshold,
            "threshold_tuning": {
                "objective": thr_objective,
                "beta": thr_beta,
                "min_recall": thr_min_recall,
                "grid_min": thr_grid_min,
                "grid_max": thr_grid_max,
                "grid_step": thr_grid_step,
            },
            "test": test_m,
            "test_thresholded": test_thr_m,
            "input_shape": loaders["input_shape"],
            "label_encoding": {
                "classes": loaders["label_encoding"].classes,
                "to_index": loaders["label_encoding"].to_index,
            },
            "dataset": loaders["metadata"],
            "note": "Phase 1 binary fall detector with train-only per-feature normalization.",
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