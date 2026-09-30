import os
import random

import numpy as np
import torch


def set_reproducibility(seed_model: int, seed_train: int, seed_split: int, seed_eval: int):
    torch.manual_seed(seed_model)
    random.seed(seed_train)
    np.random.seed(seed_split)

    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0

    for inputs, labels, _ in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)
