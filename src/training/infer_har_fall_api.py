from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, f1_score, recall_score

from src.data.har_fall_dataset import _build_windows, _split_trials, discover_trials
from src.models import CNNBiLSTMAttentionClassifier, CNNBiLSTMClassifier, CNN1DClassifier

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "har_fall_phase1.yaml"
DEFAULT_CHECKPOINT = ROOT / "outputs" / "har_fall_phase1" / "best_model.pt"
DEFAULT_METRICS = ROOT / "outputs" / "har_fall_phase1" / "metrics.json"
DEFAULT_BANK = ROOT / "outputs" / "har_fall_phase1" / "replay_bank_har.npz"
DEFAULT_NORMALIZATION = ROOT / "outputs" / "har_fall_phase1" / "normalization.json"


def _load_config(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _build_model_from_config(config: Dict[str, Any], in_channels: int, num_classes: int) -> torch.nn.Module:
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "cnn_bilstm_attention")).lower()

    if model_name == "cnn1d":
        return CNN1DClassifier(
            in_channels=int(model_cfg.get("in_channels", in_channels)),
            seq_len=256,
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


def _load_model(config: Dict[str, Any], checkpoint_path: Path) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    state = torch.load(checkpoint_path, map_location="cpu")
    input_shape = state.get("input_shape", {"channels": 270, "seq_len": 256})
    num_classes = int(state.get("num_classes", 2))

    model = _build_model_from_config(config, in_channels=int(input_shape["channels"]), num_classes=num_classes)
    model.load_state_dict(state["state_dict"])
    model.eval()
    return model, state


def _load_threshold(metrics_path: Path) -> float:
    if not metrics_path.exists():
        return 0.5
    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    return float(metrics.get("best_threshold", 0.5))


def _apply_saved_normalization(x: np.ndarray, norm_path: Path) -> np.ndarray:
    if not norm_path.exists():
        return x.astype(np.float32)

    with norm_path.open("r", encoding="utf-8") as f:
        norm = json.load(f)

    mean = np.asarray(norm["mean"], dtype=np.float32).reshape(1, -1, 1)
    std = np.asarray(norm["std"], dtype=np.float32).reshape(1, -1, 1)
    std = np.where(std < 1e-6, 1.0, std)
    return ((x - mean) / std).astype(np.float32)


def _build_or_load_replay_bank(
    config: Dict[str, Any],
    bank_path: Path,
    norm_path: Path,
    per_class: int = 256,
    max_test_trials: int = 600,
) -> Tuple[np.ndarray, np.ndarray]:
    if bank_path.exists():
        data = np.load(bank_path)
        return data["x"], data["y"]

    data_cfg = config.get("data", {})
    split_cfg = config.get("split", {})

    root_dir = Path(data_cfg.get("root_dir", "data/v38wjmz6f6-1"))
    positive_activities = data_cfg.get("positive_activities", [2, 5])
    max_files = data_cfg.get("max_files")
    window_size = int(data_cfg.get("window_size", 256))
    stride = int(data_cfg.get("stride", 128))
    min_packets = int(data_cfg.get("min_packets", 64))
    feature_mode = str(data_cfg.get("feature_mode", "amp_log_sincos")).lower()
    motion_detrend = bool(data_cfg.get("motion_detrend", False))
    temporal_diff = bool(data_cfg.get("temporal_diff", False))
    packet_step = int(data_cfg.get("packet_step", 1))
    max_windows_per_trial = int(data_cfg.get("max_windows_per_trial", 0))
    preprocess_workers = int(data_cfg.get("preprocess_workers", 8))

    trials = discover_trials(root_dir, positive_activities=positive_activities, max_files=max_files)
    _, _, test_trials = _split_trials(trials, split_cfg=split_cfg)
    if max_test_trials > 0 and len(test_trials) > max_test_trials:
        test_trials = test_trials[:max_test_trials]

    x, y = _build_windows(
        test_trials,
        window_size=window_size,
        stride=stride,
        min_packets=min_packets,
        feature_mode=feature_mode,
        motion_detrend=motion_detrend,
        temporal_diff=temporal_diff,
        packet_step=packet_step,
        max_windows_per_trial=max_windows_per_trial,
        preprocess_workers=preprocess_workers,
    )
    x = _apply_saved_normalization(x, norm_path=norm_path)
    x = x.astype(np.float16)
    y = y.astype(np.int64)

    idx_fall = np.where(y == 1)[0]
    idx_non = np.where(y == 0)[0]

    rng = np.random.default_rng(seed=42)
    rng.shuffle(idx_fall)
    rng.shuffle(idx_non)

    keep_fall = idx_fall[: min(per_class, len(idx_fall))]
    keep_non = idx_non[: min(per_class, len(idx_non))]
    keep = np.concatenate([keep_fall, keep_non])
    rng.shuffle(keep)

    x_small = x[keep]
    y_small = y[keep]

    bank_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(bank_path, x=x_small, y=y_small)
    return x_small, y_small


def _predict_batch(model: torch.nn.Module, x_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    with torch.no_grad():
        x = torch.from_numpy(x_np.astype(np.float32))
        logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        pred = np.argmax(probs, axis=1)
    return pred, probs


def infer_random_replay(model: torch.nn.Module, threshold: float, x_bank: np.ndarray, y_bank: np.ndarray, count: int) -> Dict[str, Any]:
    count = max(1, min(int(count), len(x_bank)))
    idxs = list(range(len(x_bank)))
    random.shuffle(idxs)
    pick = np.asarray(idxs[:count], dtype=np.int64)

    pred, probs = _predict_batch(model, x_bank[pick])
    p_fall = probs[:, 1]

    samples: List[Dict[str, Any]] = []
    for i in range(count):
        true_y = int(y_bank[pick[i]])
        pred_y = int(pred[i])
        # Alert if we predicted a fall (pred_y == 1)
        alert = (pred_y == 1)
        samples.append(
            {
                "sample_index": int(pick[i]),
                "true_label": "fall" if true_y == 1 else "non_fall",
                "pred_label": "fall" if pred_y == 1 else "non_fall",
                "fall_probability": float(p_fall[i]),
                "confidence": float(max(probs[i, 0], probs[i, 1])),
                "alert": bool(alert),
                "correct": bool(true_y == pred_y),
            }
        )

    y_true = y_bank[pick]
    y_pred = pred
    fall_mask = y_true == 1
    tp = int(np.sum((y_pred == 1) & fall_mask))
    actual_falls = int(np.sum(fall_mask))
    fn = int(actual_falls - tp)
    fall_recall = float(tp / actual_falls) if actual_falls > 0 else 0.0
    return {
        "mode": "random_replay",
        "threshold": float(threshold),
        "summary": {
            "count": int(count),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
            "fall_recall": fall_recall,
            "fall_tp": tp,
            "fall_fn": fn,
            "fall_actual": actual_falls,
        },
        "samples": samples,
    }


def _simulate_window(
    x: np.ndarray,
    noise_std: float,
    attenuation: float,
    phase_offset: float,
    channel_dropout: float,
    temporal_jitter: int,
) -> np.ndarray:
    out = x.astype(np.float32).copy()
    c = out.shape[0]

    rng = np.random.default_rng(seed=123)

    if attenuation > 0:
        amp_channels = c // 3
        out[:amp_channels, :] *= max(0.1, 1.0 - attenuation)

    if phase_offset != 0.0:
        amp_channels = c // 3
        phase_sin_start = amp_channels
        phase_cos_start = amp_channels * 2

        sin_part = out[phase_sin_start:phase_cos_start, :]
        cos_part = out[phase_cos_start:, :]
        out[phase_sin_start:phase_cos_start, :] = sin_part * np.cos(phase_offset) + cos_part * np.sin(phase_offset)
        out[phase_cos_start:, :] = cos_part * np.cos(phase_offset) - sin_part * np.sin(phase_offset)

    if channel_dropout > 0:
        n_drop = int(round(channel_dropout * c))
        if n_drop > 0:
            drop_idx = rng.choice(c, size=min(n_drop, c), replace=False)
            out[drop_idx, :] = 0.0

    if temporal_jitter != 0:
        shift = int(np.clip(temporal_jitter, -24, 24))
        out = np.roll(out, shift=shift, axis=1)

    if noise_std > 0:
        out += rng.normal(loc=0.0, scale=max(0.0, noise_std), size=out.shape).astype(np.float32)

    return out


def infer_simulate(
    model: torch.nn.Module,
    threshold: float,
    x_bank: np.ndarray,
    y_bank: np.ndarray,
    base_index: int,
    noise_std: float,
    attenuation: float,
    phase_offset: float,
    channel_dropout: float,
    temporal_jitter: int,
) -> Dict[str, Any]:
    idx = int(base_index) % len(x_bank)
    x0 = x_bank[idx]
    y0 = int(y_bank[idx])

    pred_base, probs_base = _predict_batch(model, x0[None, ...])
    p_base = float(probs_base[0, 1])

    x_sim = _simulate_window(
        x=x0,
        noise_std=float(noise_std),
        attenuation=float(attenuation),
        phase_offset=float(phase_offset),
        channel_dropout=float(channel_dropout),
        temporal_jitter=int(temporal_jitter),
    )

    pred_sim, probs_sim = _predict_batch(model, x_sim[None, ...])
    p_sim = float(probs_sim[0, 1])

    return {
        "mode": "simulate",
        "threshold": float(threshold),
        "base_index": int(idx),
        "base": {
            "true_label": "fall" if y0 == 1 else "non_fall",
            "pred_label": "fall" if int(pred_base[0]) == 1 else "non_fall",
            "fall_probability": p_base,
            "alert": bool(int(pred_base[0]) == 1),
        },
        "simulated": {
            "pred_label": "fall" if int(pred_sim[0]) == 1 else "non_fall",
            "fall_probability": p_sim,
            "alert": bool(int(pred_sim[0]) == 1),
            "confidence": float(max(probs_sim[0, 0], probs_sim[0, 1])),
        },
        "delta": {
            "fall_probability_delta": float(p_sim - p_base),
            "flipped": bool(int(pred_base[0]) != int(pred_sim[0])),
        },
        "simulation": {
            "noise_std": float(noise_std),
            "attenuation": float(attenuation),
            "phase_offset": float(phase_offset),
            "channel_dropout": float(channel_dropout),
            "temporal_jitter": int(temporal_jitter),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="HAR fall inference API")
    parser.add_argument("--mode", type=str, default="random_replay", choices=["random_replay", "simulate"])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--json-stdin", action="store_true")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", type=str, default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--metrics", type=str, default=str(DEFAULT_METRICS))
    parser.add_argument("--bank", type=str, default=str(DEFAULT_BANK))
    parser.add_argument("--normalization", type=str, default=str(DEFAULT_NORMALIZATION))
    args = parser.parse_args()

    payload: Dict[str, Any] = {}
    if args.json_stdin:
        raw = input().strip()
        payload = json.loads(raw) if raw else {}

    config = _load_config(Path(args.config))
    model, state = _load_model(config=config, checkpoint_path=Path(args.checkpoint))
    threshold = _load_threshold(Path(args.metrics))
    x_bank, y_bank = _build_or_load_replay_bank(
        config=config,
        bank_path=Path(args.bank),
        norm_path=Path(args.normalization),
    )

    if args.mode == "random_replay":
        req_count = int(payload.get("count", args.count))
        out = infer_random_replay(model=model, threshold=threshold, x_bank=x_bank, y_bank=y_bank, count=req_count)
    else:
        out = infer_simulate(
            model=model,
            threshold=threshold,
            x_bank=x_bank,
            y_bank=y_bank,
            base_index=int(payload.get("base_index", 0)),
            noise_std=float(payload.get("noise_std", 0.03)),
            attenuation=float(payload.get("attenuation", 0.1)),
            phase_offset=float(payload.get("phase_offset", 0.02)),
            channel_dropout=float(payload.get("channel_dropout", 0.05)),
            temporal_jitter=int(payload.get("temporal_jitter", 0)),
        )

    model_cfg = config.get("model", {})
    out["model"] = str(model_cfg.get("name", "cnn_bilstm_attention"))
    out["model_tag"] = "gamma2.5"
    out["input_shape"] = state.get("input_shape", {"channels": 270, "seq_len": 256})
    print(json.dumps(out))


if __name__ == "__main__":
    main()
