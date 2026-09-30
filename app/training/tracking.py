import mlflow
import torch
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

    avg_loss = total_loss / len(dataloader)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)

    return avg_loss, acc, macro_f1


def train_with_mlflow(
    config,
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion,
    device,
    experiment_name="P3_Campaign",
):
    mlflow.set_experiment(experiment_name)
    early_stopping = EarlyStopping(patience=config.patience, min_delta=config.min_delta)

    with mlflow.start_run() as run:
        mlflow.log_params(config.model_dump())

        stopped_epoch = config.max_epochs

        for epoch in range(1, config.max_epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            _, train_acc, _ = evaluate_epoch(model, train_loader, criterion, device)

            val_loss, val_acc, val_macro_f1 = evaluate_epoch(model, val_loader, criterion, device)

            mlflow.log_metrics(
                {
                    "train_loss": train_loss,
                    "train_accuracy": train_acc,
                    "val_loss": val_loss,
                    "val_accuracy": val_acc,
                    "val_macro_f1": val_macro_f1,
                },
                step=epoch,
            )

            early_stopping(val_loss, model, epoch)
            if early_stopping.early_stop:
                stopped_epoch = epoch
                break

        model.load_state_dict(early_stopping.best_weights)
        mlflow.log_metric("best_epoch", early_stopping.best_epoch)
        mlflow.log_metric("stopped_epoch", stopped_epoch)

        return run.info.run_id
