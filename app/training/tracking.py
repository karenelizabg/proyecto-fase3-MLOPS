import hashlib
import json
import os
import platform
import re
import subprocess
import tempfile
import time

import mlflow
import pandas as pd
import torch
import torchvision
from sklearn.metrics import accuracy_score, f1_score
from training.trainer import EarlyStopping, train_epoch


def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, targets, _ in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    avg_loss = total_loss / len(dataloader) if len(dataloader) > 0 else float("inf")
    acc = accuracy_score(all_targets, all_preds) if all_targets else 0.0
    macro_f1 = (
        f1_score(all_targets, all_preds, average="macro", zero_division=0) if all_targets else 0.0
    )

    return avg_loss, acc, macro_f1


def get_git_info():
    commit = os.getenv("GIT_COMMIT")
    if commit:
        return commit, os.getenv("GIT_DIRTY", "false").lower()

    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
        )
        is_dirty = (
            subprocess.check_output(["git", "status", "--porcelain"], stderr=subprocess.DEVNULL)
            .decode("ascii")
            .strip()
            != ""
        )
        return commit, "true" if is_dirty else "false"
    except Exception:
        return "unknown", "true"


def compute_sha256(filepath):
    hasher = hashlib.sha256()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def train_with_mlflow(
    config, model, train_loader, val_loader, optimizer, criterion, device, tags=None
):
    if tags is None:
        tags = {}

    run_kind = tags.get("run_kind", "")
    grid_row = tags.get("grid_row", "")

    if run_kind not in {"smoke", "campaign"}:
        raise ValueError(f"run_kind debe ser 'smoke' o 'campaign', se recibió '{run_kind}'")
    if run_kind == "smoke" and grid_row != "":
        raise ValueError("grid_row debe estar vacío para run_kind 'smoke'")
    if run_kind == "campaign" and not re.match(r"^r(0[1-9]|1[0-2])$", grid_row):
        raise ValueError(f"grid_row inválido para campaign: {grid_row}")

    mlflow.set_experiment("clasificador-perro-gato")
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)
    start_time = time.time()
    curves_data = []

    commit, dirty = get_git_info()
    tags.update({"git_commit": commit, "git_dirty": dirty})

    meta_path = os.getenv("MANIFEST_META_PATH", "reports/manifests/v0.1.1/manifest_meta.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Archivo de metadatos no encontrado: {meta_path}")

    with open(meta_path, "r") as f:
        meta = json.load(f)
        tags["manifest_id"] = meta.get("manifest_id")
        tags["manifest_sha256"] = meta.get("manifest_sha256")
        tags["classes"] = json.dumps(meta.get("classes", {}))
        tags["release"] = meta.get("release", {}).get("name", "unknown")

    tags["python_version"] = platform.python_version()
    tags["torch_version"] = torch.__version__
    tags["torchvision_version"] = torchvision.__version__
    tags["platform"] = platform.platform()
    tags["device"] = str(device)

    if getattr(config, "seed_aug", None) is not None:
        torch.manual_seed(config.seed_aug)

    with mlflow.start_run() as run:
        mlflow.log_params(config.model_dump())
        mlflow.set_tags(tags)

        stopped_epoch = config.max_epochs

        for epoch in range(1, config.max_epochs + 1):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, val_macro_f1 = evaluate_epoch(model, val_loader, criterion, device)

            metrics = {
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "val_macro_f1": val_macro_f1,
            }
            mlflow.log_metrics(metrics, step=epoch)
            curves_data.append(metrics)

            early_stopping(val_loss, val_acc, val_macro_f1, model, epoch)
            if early_stopping.early_stop:
                stopped_epoch = epoch
                break

        mlflow.log_metric("duration_s", time.time() - start_time)

        with tempfile.TemporaryDirectory() as tmpdir:
            curves_path = os.path.join(tmpdir, "curves.csv")
            pd.DataFrame(curves_data).to_csv(curves_path, index=False)
            mlflow.log_artifact(curves_path)

            if early_stopping.best_weights is not None:
                model.load_state_dict(early_stopping.best_weights)
                best_pt_path = os.path.join(tmpdir, "best.pt")
                torch.save(early_stopping.best_weights, best_pt_path)
                mlflow.log_artifact(best_pt_path, artifact_path="checkpoint")

                mlflow.set_tag("checkpoint_sha256", compute_sha256(best_pt_path))

        mlflow.log_metric("best_epoch", early_stopping.best_epoch)
        mlflow.log_metric("best_val_loss", early_stopping.best_loss)
        mlflow.log_metric("best_val_accuracy", early_stopping.best_val_accuracy)
        mlflow.log_metric("best_val_macro_f1", early_stopping.best_val_macro_f1)
        mlflow.log_metric("stopped_epoch", stopped_epoch)

        return run.info.run_id
