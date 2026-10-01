import multiprocessing
import os
import sys
import time

import mlflow
import torch
import torch.nn as nn
from training.config import TrainingConfig
from training.data import create_dataloader
from training.model import build_model
from training.tracking import train_with_mlflow

app_dir = os.path.abspath(os.path.dirname(__file__))
project_root = os.path.dirname(app_dir)
sys.path.insert(0, app_dir)

os.environ["MANIFEST_META_PATH"] = os.path.join(
    project_root, "reports", "manifests", "v0.1.1", "manifest_meta.json"
)
os.environ["GIT_COMMIT"] = "smoke-test-commit"
os.environ["GIT_DIRTY"] = "false"


def run_training_cycle(run_name):
    print(f"\n--- Iniciando {run_name} ---")
    device = torch.device("cpu")

    config = TrainingConfig(
        optimizer="adam",
        batch_size=16,
        max_epochs=15,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=0,
        dropout=0.0,
        seed_split=42,
        seed_train=43,
        seed_aug=44,
        seed_model=45,
        patience=2,
        min_delta=0.01,
    )

    model = build_model(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    criterion = nn.CrossEntropyLoss()

    loader_args = {
        "manifest_path": os.path.join(app_dir, "manifest.csv"),
        "base_dir": app_dir,
        "batch_size": 2,
        "image_size": config.image_size,
        "seed_train": config.seed_train,
        "num_workers": 0,
    }

    train_loader = create_dataloader(split="train", **loader_args)
    val_loader = create_dataloader(split="val", **loader_args)

    tags = {"run_kind": "smoke", "grid_row": ""}

    start_time = time.time()
    run_id = train_with_mlflow(
        config, model, train_loader, val_loader, optimizer, criterion, device, tags=tags
    )
    duration = time.time() - start_time

    run = mlflow.get_run(run_id)
    print(f"{run_name} completado en {duration:.2f}s. Run ID: {run_id}")
    return run_id, run.data.metrics, duration


def verify_checkpoint_isolated(run_id):
    """Ejecutado en un proceso nuevo para garantizar memoria limpia"""
    import torch
    from mlflow.artifacts import download_artifacts
    from training.config import TrainingConfig
    from training.model import build_model

    print("\n--- Verificando Checkpoint en Proceso Nuevo ---")
    config = TrainingConfig(
        optimizer="adam",
        batch_size=16,
        max_epochs=15,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=0,
        dropout=0.0,
        seed_split=42,
        seed_train=43,
        seed_aug=44,
        seed_model=45,
        patience=2,
        min_delta=0.01,
    )

    model = build_model(config)
    ckpt_path = download_artifacts(run_id=run_id, artifact_path="checkpoint/best.pt")

    model.load_state_dict(torch.load(ckpt_path, weights_only=True))
    model.eval()

    dummy_input = torch.ones(1, 3, 128, 128)
    with torch.no_grad():
        output = model(dummy_input)

    print(f"Predicción reproducida con éxito. Tensor de salida: {output}")


if __name__ == "__main__":
    run_id_1, metrics_1, t1 = run_training_cycle("Corrida 1")
    run_id_2, metrics_2, t2 = run_training_cycle("Corrida 2")

    print("\n--- Resultados del Determinismo ---")
    keys_to_check = ["best_val_loss", "best_val_accuracy", "best_val_macro_f1"]
    for k in keys_to_check:
        v1, v2 = metrics_1[k], metrics_2[k]
        diff = abs(v1 - v2)
        print(f"{k}: Corrida1={v1:.4f} | Corrida2={v2:.4f} | Diferencia={diff:.4f}")
        assert diff < 1e-6, f"Fallo de determinismo en {k}"

    print("\n¡Las métricas coinciden perfectamente!")

    p = multiprocessing.Process(target=verify_checkpoint_isolated, args=(run_id_1,))
    p.start()
    p.join()

    assert p.exitcode == 0, "Fallo al cargar el checkpoint en un proceso nuevo."
    print(f"\nSmoke test completado exitosamente. Tiempos - C1: {t1:.1f}s, C2: {t2:.1f}s")
