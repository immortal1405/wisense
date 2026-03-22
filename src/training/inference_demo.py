from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from src.data.csi_dataset import build_dataloaders
from src.models import CNN1DClassifier, CNNBiLSTMClassifier


def _load_yaml(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model(model_cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
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


def _evaluate_model(
    model: torch.nn.Module,
    loader,
    class_names: List[str],
    device: torch.device,
    sample_count: int = 12,
) -> Tuple[Dict[str, float], np.ndarray, List[Dict[str, Any]]]:
    model.eval()
    y_true, y_pred = [], []
    sample_rows: List[Dict[str, Any]] = []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(y.cpu().numpy().tolist())
            y_pred.extend(preds.cpu().numpy().tolist())

            if len(sample_rows) < sample_count:
                confs = torch.max(probs, dim=1).values
                for i in range(x.shape[0]):
                    if len(sample_rows) >= sample_count:
                        break
                    ti = int(y[i].item())
                    pi = int(preds[i].item())
                    sample_rows.append(
                        {
                            "true_idx": ti,
                            "pred_idx": pi,
                            "true_label": class_names[ti],
                            "pred_label": class_names[pi],
                            "confidence": float(confs[i].item()),
                            "correct": bool(ti == pi),
                        }
                    )

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics = {
        "accuracy": float(accuracy_score(y_true_arr, y_pred_arr)),
        "macro_f1": float(f1_score(y_true_arr, y_pred_arr, average="macro")),
    }
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=np.arange(len(class_names)))
    return metrics, cm, sample_rows


def _plot_metric_bars(metrics_by_model: Dict[str, Dict[str, float]], output_path: Path) -> None:
    sns.set_theme(style="whitegrid")
    models = list(metrics_by_model.keys())
    acc = [metrics_by_model[m]["accuracy"] for m in models]
    f1 = [metrics_by_model[m]["macro_f1"] for m in models]

    x = np.arange(len(models))
    w = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - w / 2, acc, width=w, label="Accuracy", color="#0f766e")
    bars2 = ax.bar(x + w / 2, f1, width=w, label="Macro-F1", color="#ca8a04")

    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Score")
    ax.set_title("Wi-Sense Model Comparison (Standard Split)")
    ax.legend(loc="upper left")

    for bars in (bars1, bars2):
        for b in bars:
            h = b.get_height()
            ax.annotate(f"{h:.3f}", (b.get_x() + b.get_width() / 2, h), ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_confusion(cm: np.ndarray, class_names: List[str], title: str, output_path: Path) -> None:
    sns.set_theme(style="white")
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", cbar=False, xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    workspace = Path(".")
    output_dir = workspace / "outputs" / "presentation"
    output_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = _load_yaml(workspace / "configs" / "base.yaml")
    bilstm_cfg = _load_yaml(workspace / "configs" / "model_cnn_bilstm.yaml")

    loaders_bundle = build_dataloaders(base_cfg)
    test_loader = loaders_bundle["test_loader"]
    class_names = loaders_bundle["label_encoding"].classes

    device = torch.device("cpu")

    model_specs = [
        {
            "name": "CNN1D",
            "cfg": base_cfg,
            "weights": workspace / "outputs_modal" / "cnn1d_type_baseline" / "best_model.pt",
        },
        {
            "name": "CNN-BiLSTM",
            "cfg": bilstm_cfg,
            "weights": workspace / "outputs_modal" / "cnn_bilstm_type_baseline" / "best_model.pt",
        },
    ]

    inference_report: Dict[str, Any] = {
        "class_names": class_names,
        "models": {},
    }

    for spec in model_specs:
        model = _build_model(spec["cfg"]["model"], num_classes=len(class_names))
        state_dict = torch.load(spec["weights"], map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)

        metrics, cm, samples = _evaluate_model(model, test_loader, class_names, device)
        inference_report["models"][spec["name"]] = {
            "metrics": metrics,
            "samples": samples,
        }

        _plot_confusion(
            cm=cm,
            class_names=class_names,
            title=f"{spec['name']} Confusion Matrix",
            output_path=output_dir / f"confusion_{spec['name'].lower().replace('-', '_')}.png",
        )

    # Prefer canonical Modal metrics if present for headline comparison chart.
    canonical_path = workspace / "modal_results_standard.json"
    if canonical_path.exists():
        with canonical_path.open("r", encoding="utf-8") as f:
            canonical = json.load(f)
        chart_metrics = {
            "CNN1D": {
                "accuracy": float(canonical["runs"]["cnn1d_type_baseline"]["test_accuracy"]),
                "macro_f1": float(canonical["runs"]["cnn1d_type_baseline"]["test_macro_f1"]),
            },
            "CNN-BiLSTM": {
                "accuracy": float(canonical["runs"]["cnn_bilstm_type_baseline"]["test_accuracy"]),
                "macro_f1": float(canonical["runs"]["cnn_bilstm_type_baseline"]["test_macro_f1"]),
            },
        }
    else:
        chart_metrics = {
            k: v["metrics"] for k, v in inference_report["models"].items()
        }

    _plot_metric_bars(chart_metrics, output_dir / "model_comparison_standard.png")

    with (output_dir / "inference_report.json").open("w", encoding="utf-8") as f:
        json.dump(inference_report, f, indent=2)

    print("Saved presentation assets to:", output_dir)
    print("- model_comparison_standard.png")
    print("- confusion_cnn1d.png")
    print("- confusion_cnn_bilstm.png")
    print("- inference_report.json")


if __name__ == "__main__":
    main()
