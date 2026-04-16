from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path

import modal

APP_NAME = "wisense-csi"
REMOTE_ROOT = "/root/wisense"
ARTIFACT_ROOT = "/artifacts"

app = modal.App(APP_NAME)
artifacts_volume = modal.Volume.from_name("wisense-artifacts", create_if_missing=True)

# CUDA-enabled base image for H100 runs.
image = (
    modal.Image.from_registry("pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime")
    .apt_install("git")
    .pip_install(
        "numpy>=1.24",
        "pandas>=2.0",
        "scikit-learn>=1.3",
        "PyYAML>=6.0",
        "tqdm>=4.66",
        "matplotlib>=3.8",
        "seaborn>=0.13",
    )
    .add_local_dir("src", remote_path=f"{REMOTE_ROOT}/src")
    .add_local_dir("configs", remote_path=f"{REMOTE_ROOT}/configs")
    .add_local_dir("data", remote_path=f"{REMOTE_ROOT}/data")
)


@app.function(
    image=image,
    gpu="H100",
    cpu=8,
    memory=32768,
    timeout=60 * 60,
    volumes={ARTIFACT_ROOT: artifacts_volume},
)
def train(
    config_name: str = "base.yaml",
    module_name: str = "src.training.train",
    epochs: int = 20,
    max_batches: int = 0,
) -> None:
    project_dir = Path(REMOTE_ROOT)
    config_path = project_dir / "configs" / config_name

    cmd = [
        "python",
        "-u",
        "-m",
        module_name,
        "--config",
        str(config_path),
        "--epochs",
        str(epochs),
    ]
    if max_batches > 0:
        cmd.extend(["--max-batches", str(max_batches)])

    subprocess.run(cmd, cwd=str(project_dir), check=True)

    artifact_outputs = Path(ARTIFACT_ROOT) / "outputs"
    artifact_outputs.mkdir(parents=True, exist_ok=True)
    shutil.copytree(project_dir / "outputs", artifact_outputs, dirs_exist_ok=True)

    if hasattr(artifacts_volume, "commit"):
        artifacts_volume.commit()


@app.function(image=image, volumes={ARTIFACT_ROOT: artifacts_volume})
def download_run_file(run_name: str, file_name: str) -> bytes:
    src = Path(ARTIFACT_ROOT) / "outputs" / run_name / file_name
    if not src.exists():
        raise FileNotFoundError(f"Artifact not found: {src}")
    return src.read_bytes()


@app.function(image=image, volumes={ARTIFACT_ROOT: artifacts_volume})
def list_saved_runs() -> list[str]:
    outputs_dir = Path(ARTIFACT_ROOT) / "outputs"
    if not outputs_dir.exists():
        return []
    return sorted([p.name for p in outputs_dir.iterdir() if p.is_dir()])


@app.local_entrypoint()
def main(
    action: str = "train",
    config_name: str = "base.yaml",
    module_name: str = "src.training.train",
    epochs: int = 20,
    max_batches: int = 0,
    run_name: str = "",
    file_name: str = "best_model.pt",
    local_dir: str = "outputs_modal",
) -> None:
    if action == "train":
        train.remote(
            config_name=config_name,
            module_name=module_name,
            epochs=epochs,
            max_batches=max_batches,
        )
        return

    if action == "list":
        runs = list_saved_runs.remote()
        print(json.dumps({"runs": runs}, indent=2))
        return

    if action == "download":
        if not run_name:
            raise ValueError("run_name is required when action=download")

        data = download_run_file.remote(run_name=run_name, file_name=file_name)
        out_dir = Path(local_dir) / run_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / file_name
        out_path.write_bytes(data)
        print(f"Saved artifact to: {out_path}")
        return

    raise ValueError(f"Unsupported action: {action}")
