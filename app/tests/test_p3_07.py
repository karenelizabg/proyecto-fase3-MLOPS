import math

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def get_dummy_dataloader(num_samples=10, batch_size=3, seed=42):
    X = torch.randn(num_samples, 3, 224, 224)
    y = torch.randint(0, 2, (num_samples,))
    ids = torch.arange(num_samples)
    dataset = TensorDataset(X, y, ids)

    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)


def test_same_crop_id_order_with_same_seed():
    loader1 = get_dummy_dataloader(seed=42)
    loader2 = get_dummy_dataloader(seed=42)

    _, _, ids1 = next(iter(loader1))
    _, _, ids2 = next(iter(loader2))

    assert torch.equal(ids1, ids2)


def test_ceil_steps_per_epoch():
    num_samples = 10
    batch_size = 3
    loader = get_dummy_dataloader(num_samples=num_samples, batch_size=batch_size)

    expected_steps = math.ceil(num_samples / batch_size)
    actual_steps = len(loader)

    assert actual_steps == expected_steps


def test_weights_change_after_short_run():
    model = nn.Linear(10, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()

    initial_weights = model.weight.clone()

    x = torch.randn(4, 10)
    y = torch.randint(0, 2, (4,))

    optimizer.zero_grad()
    loss = criterion(model(x), y)
    loss.backward()
    optimizer.step()

    assert not torch.equal(initial_weights, model.weight)
