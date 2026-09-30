import mlflow
import torch
import torch.nn as nn
from training.config import TrainingConfig
from training.data import create_dataloader
from training.model import build_model
from training.tracking import train_with_mlflow
from training.trainer import EarlyStopping


def test_seed_aug_workers_effect(mock_dataset_env):
    manifest_path, base_dir = mock_dataset_env

    def get_batch(seed_aug_val, nw_val):
        loader = create_dataloader(
            manifest_path,
            base_dir,
            "train",
            batch_size=2,
            image_size=128,
            seed_train=43,
            seed_aug=seed_aug_val,
            num_workers=nw_val,
        )
        return next(iter(loader))

    for nw in [0, 2]:
        t1, _, crop1 = get_batch(44, nw)
        t2, _, crop2 = get_batch(44, nw)

        assert torch.equal(t1, t2), (
            f"Fallo con num_workers={nw}: Tensores distintos con misma semilla"
        )
        assert torch.equal(crop1, crop2)

        t3, _, crop3 = get_batch(99, nw)

        assert not torch.equal(t1, t3), (
            f"Fallo con num_workers={nw}: Tensores idénticos con distinta semilla"
        )
        assert torch.equal(crop1, crop3), f"Fallo con num_workers={nw}: El orden cambió"


def test_early_stopping_controlled_sequence():
    early_stopping = EarlyStopping(patience=3, min_delta=0.01)

    model = nn.Linear(10, 2)

    val_losses = [0.90, 0.70, 0.60, 0.5995, 0.62, 0.61]
    saved_weights_at_epoch_3 = None

    for epoch, loss in enumerate(val_losses, start=1):
        nn.init.uniform_(model.weight)

        if epoch == 3:
            saved_weights_at_epoch_3 = {k: v.clone() for k, v in model.state_dict().items()}

        early_stopping(loss, model, epoch)

        if early_stopping.early_stop:
            assert epoch == 6, f"Se detuvo prematuramente en la época {epoch}"
            break

    assert early_stopping.early_stop, "No se activó el early stopping"
    assert early_stopping.best_epoch == 3, "No registró la mejor época correctamente"

    model.load_state_dict(early_stopping.best_weights)
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], saved_weights_at_epoch_3[key]), (
            "Los pesos restaurados no coinciden"
        )


def test_mlflow_contract(mock_dataset_env, tmp_path):
    manifest_path, base_dir = mock_dataset_env
    device = torch.device("cpu")

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/mlflow.db")

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
        "manifest_path": manifest_path,
        "base_dir": base_dir,
        "batch_size": 2,
        "image_size": config.image_size,
        "seed_train": config.seed_train,
        "num_workers": 0,
    }

    train_loader = create_dataloader(split="train", **loader_args)
    val_loader = create_dataloader(split="val", **loader_args)

    run_id = train_with_mlflow(
        config, model, train_loader, val_loader, optimizer, criterion, device, "Test_Experiment"
    )

    run = mlflow.get_run(run_id)
    metrics = run.data.metrics
    params = run.data.params

    assert "optimizer" in params
    assert params["max_epochs"] == "15"

    assert "val_macro_f1" in metrics, "Falta registrar val_macro_f1"
    assert "train_accuracy" in metrics, "Falta registrar train_accuracy"
    assert "best_epoch" in metrics, "Falta registrar best_epoch"
    assert "stopped_epoch" in metrics, "Falta registrar stopped_epoch"

    assert metrics["best_epoch"] > 0
    assert metrics["stopped_epoch"] <= 15
