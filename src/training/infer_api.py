from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import yaml

from src.data.csi_dataset import build_dataloaders
from src.models import CNN1DClassifier, CNNBiLSTMClassifier


def build_model(model_name: str, num_classes: int) -> torch.nn.Module:
    if model_name == "cnn1d":
        return CNN1DClassifier(num_classes=num_classes)
    if model_name == "cnn_bilstm":
        return CNNBiLSTMClassifier(num_classes=num_classes)
    raise ValueError(f"Unsupported model: {model_name}")


def _load_cfg_and_bundle(model_name: str) -> Tuple[Dict[str, Any], Dict[str, Any], torch.nn.Module, List[str]]:
    root = Path(".")
    config_path = root / "configs" / ("base.yaml" if model_name == "cnn1d" else "model_cnn_bilstm.yaml")
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    bundle = build_dataloaders(cfg)
    class_names = bundle["label_encoding"].classes
    model = build_model(model_name=model_name, num_classes=bundle["num_classes"])

    weights = (
        root
        / "outputs_modal"
        / ("cnn1d_type_baseline" if model_name == "cnn1d" else "cnn_bilstm_type_baseline")
        / "best_model.pt"
    )
    state = torch.load(weights, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    return cfg, bundle, model, class_names


def _predict_single(model: torch.nn.Module, x: torch.Tensor, class_names: List[str]) -> Dict[str, Any]:
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(torch.argmax(probs, dim=1)[0].item())
        confidence = float(probs[0, pred_idx].item())
    return {
        "pred_idx": pred_idx,
        "pred_label": class_names[pred_idx],
        "confidence": confidence,
        "probs": probs[0].detach().cpu().numpy().tolist(),
    }


def _subcarrier_importance(model: torch.nn.Module, x: torch.Tensor, pred_idx: int) -> List[Dict[str, Any]]:
    # Simple perturbation importance: zero a subcarrier pair (amp+phase) and observe confidence drop.
    with torch.no_grad():
        base_logits = model(x)
        base_probs = torch.softmax(base_logits, dim=1)
        base_conf = float(base_probs[0, pred_idx].item())

        out: List[Dict[str, Any]] = []
        for i in range(x.shape[-1]):
            x_mod = x.clone()
            x_mod[0, 0, i] = 0.0
            x_mod[0, 1, i] = 0.0
            mod_probs = torch.softmax(model(x_mod), dim=1)
            mod_conf = float(mod_probs[0, pred_idx].item())
            out.append(
                {
                    "subcarrier": int(i),
                    "importance": float(base_conf - mod_conf),
                    "base_confidence": base_conf,
                    "perturbed_confidence": mod_conf,
                }
            )

    out.sort(key=lambda z: z["importance"], reverse=True)
    return out[:8]


def _parse_vector(values: Any, expected: int = 52) -> List[float]:
    if not isinstance(values, list):
        raise ValueError("Vector must be a JSON list")
    if len(values) != expected:
        raise ValueError(f"Vector length must be {expected}")
    parsed = [float(v) for v in values]
    # Keep demo input in normalized range.
    return [max(0.0, min(1.0, v)) for v in parsed]


def _build_single_input(amp: List[float], phase: List[float]) -> torch.Tensor:
    arr = np.zeros((1, 2, 52), dtype=np.float32)
    arr[0, 0, :] = np.asarray(amp, dtype=np.float32)
    arr[0, 1, :] = np.asarray(phase, dtype=np.float32)
    return torch.from_numpy(arr)


def _parse_csv_row(csv_text: str) -> Tuple[List[float], List[float]]:
    lines = [ln.strip() for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        raise ValueError("CSV text is empty")

    first_parts = [p.strip() for p in lines[0].split(",")]
    has_header = any(p.startswith("amp_") or p.startswith("phase_") for p in first_parts)

    if has_header:
        if len(lines) < 2:
            raise ValueError("CSV with header must include at least one data row")
        header = first_parts
        row = [p.strip() for p in lines[1].split(",")]
        if len(row) != len(header):
            raise ValueError("CSV row length does not match header")
        mapping = {h: row[i] for i, h in enumerate(header)}
        amp = [float(mapping[f"amp_{i}"]) for i in range(52)]
        phase = [float(mapping[f"phase_{i}"]) for i in range(52)]
        return _parse_vector(amp), _parse_vector(phase)

    # No header path: expect exactly 104 numeric values in one line.
    row_values = [float(p) for p in first_parts]
    if len(row_values) != 104:
        raise ValueError("CSV without header must have exactly 104 values: 52 amp + 52 phase")
    amp = row_values[:52]
    phase = row_values[52:]
    return _parse_vector(amp), _parse_vector(phase)


def infer_random(model_name: str, count: int) -> Dict[str, Any]:
    _, bundle, model, class_names = _load_cfg_and_bundle(model_name)
    test_loader = bundle["test_loader"]

    # Collect a few full batches to sample from.
    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
        if len(xs) >= 3:
            break

    x_cat = torch.cat(xs, dim=0)
    y_cat = torch.cat(ys, dim=0)

    idxs = list(range(len(x_cat)))
    random.shuffle(idxs)
    idxs = idxs[: max(1, min(count, len(idxs)))]

    with torch.no_grad():
        logits = model(x_cat[idxs])
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

    samples = []
    for i, idx in enumerate(idxs):
        true_idx = int(y_cat[idx].item())
        pred_idx = int(preds[i].item())
        conf = float(torch.max(probs[i]).item())
        samples.append(
            {
                "sample_index": int(idx),
                "true_label": class_names[true_idx],
                "pred_label": class_names[pred_idx],
                "confidence": conf,
                "correct": true_idx == pred_idx,
            }
        )

    accuracy = sum(1 for s in samples if s["correct"]) / len(samples)

    return {
        "mode": "random",
        "model": model_name,
        "count": len(samples),
        "window_accuracy": accuracy,
        "samples": samples,
    }


def infer_manual(model_name: str, amp: List[float], phase: List[float]) -> Dict[str, Any]:
    _, _, model, class_names = _load_cfg_and_bundle(model_name)
    x = _build_single_input(amp, phase)
    pred = _predict_single(model, x, class_names)
    top_subcarriers = _subcarrier_importance(model, x, pred["pred_idx"])

    return {
        "mode": "manual",
        "model": model_name,
        "prediction": {
            "label": pred["pred_label"],
            "confidence": pred["confidence"],
        },
        "class_probabilities": [
            {"label": class_names[i], "probability": float(pred["probs"][i])} for i in range(len(class_names))
        ],
        "input_signal": {
            "amp": amp,
            "phase": phase,
        },
        "explanation": {
            "method": "subcarrier perturbation",
            "top_subcarriers": top_subcarriers,
        },
    }


def infer_csv(model_name: str, csv_text: str) -> Dict[str, Any]:
    amp, phase = _parse_csv_row(csv_text)
    out = infer_manual(model_name=model_name, amp=amp, phase=phase)
    out["mode"] = "csv"
    return out


def infer_simulate(
    model_name: str,
    base_index: int,
    noise: float,
    phase_offset: float,
    attenuation: float,
) -> Dict[str, Any]:
    _, bundle, model, class_names = _load_cfg_and_bundle(model_name)
    test_loader = bundle["test_loader"]

    xs, ys = [], []
    for x, y in test_loader:
        xs.append(x)
        ys.append(y)
        if len(xs) >= 3:
            break

    x_cat = torch.cat(xs, dim=0)
    y_cat = torch.cat(ys, dim=0)
    idx = int(base_index) % len(x_cat)
    x0 = x_cat[idx : idx + 1].clone()
    y0 = int(y_cat[idx].item())

    amp = x0[0, 0, :].numpy().astype(np.float32)
    phase = x0[0, 1, :].numpy().astype(np.float32)

    rng = np.random.default_rng(seed=42)
    noise = max(0.0, float(noise))
    attenuation = max(0.0, min(0.9, float(attenuation)))
    phase_offset = float(phase_offset)

    amp_mod = np.clip(amp * (1.0 - attenuation) + noise * rng.standard_normal(amp.shape), 0.0, 1.0)
    phase_mod = np.clip(phase + phase_offset + noise * rng.standard_normal(phase.shape), 0.0, 1.0)

    x_mod = _build_single_input(amp_mod.tolist(), phase_mod.tolist())
    pred = _predict_single(model, x_mod, class_names)
    top_subcarriers = _subcarrier_importance(model, x_mod, pred["pred_idx"])

    return {
        "mode": "simulate",
        "model": model_name,
        "base_index": idx,
        "true_label": class_names[y0],
        "simulation": {
            "noise": noise,
            "phase_offset": phase_offset,
            "attenuation": attenuation,
        },
        "prediction": {
            "label": pred["pred_label"],
            "confidence": pred["confidence"],
            "correct": pred["pred_idx"] == y0,
        },
        "input_signal": {
            "amp": amp_mod.tolist(),
            "phase": phase_mod.tolist(),
        },
        "explanation": {
            "method": "subcarrier perturbation",
            "top_subcarriers": top_subcarriers,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wi-Sense inference API")
    parser.add_argument("--model", type=str, default="cnn_bilstm", choices=["cnn1d", "cnn_bilstm"])
    parser.add_argument("--mode", type=str, default="random", choices=["random", "manual", "csv", "simulate"])
    parser.add_argument("--count", type=int, default=8)
    parser.add_argument("--json-stdin", action="store_true")
    args = parser.parse_args()

    payload: Dict[str, Any] = {}
    if args.json_stdin:
        raw = input().strip()
        payload = json.loads(raw) if raw else {}

    if args.mode == "random":
        result = infer_random(model_name=args.model, count=args.count)
    elif args.mode == "manual":
        amp = _parse_vector(payload.get("amp", []))
        phase = _parse_vector(payload.get("phase", []))
        result = infer_manual(model_name=args.model, amp=amp, phase=phase)
    elif args.mode == "csv":
        csv_text = str(payload.get("csv_text", ""))
        result = infer_csv(model_name=args.model, csv_text=csv_text)
    else:
        result = infer_simulate(
            model_name=args.model,
            base_index=int(payload.get("base_index", 0)),
            noise=float(payload.get("noise", 0.05)),
            phase_offset=float(payload.get("phase_offset", 0.02)),
            attenuation=float(payload.get("attenuation", 0.1)),
        )

    print(json.dumps(result))


if __name__ == "__main__":
    main()
