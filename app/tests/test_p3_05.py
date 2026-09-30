import numpy as np
import pytest
import torch
from PIL import Image
from pydantic import ValidationError
from training.config import TrainingConfig
from training.model import build_model
from training.preprocess import get_preprocessing_transforms


def test_config_invalid_values_rejected_with_field_name():
    with pytest.raises(ValidationError) as exc_info:
        TrainingConfig(
            optimizer="Adam",
            batch_size=32,
            max_epochs=10,
            learning_rate=0.001,
            image_size=224,
            hidden_layers=128,
            dropout=1.5,
            seed_split=42,
            seed_train=42,
            seed_model=42,
            seed_eval=42,
            patience=5,
            min_delta=0.01,
        )
    assert "dropout" in str(exc_info.value).lower()


def test_model_output_shape_and_frozen_layers():
    config = TrainingConfig(
        optimizer="adam",
        batch_size=16,
        max_epochs=15,
        learning_rate=0.001,
        image_size=128,
        hidden_layers=1,
        dropout=0.3,
        seed_split=42,
        seed_train=43,
        seed_aug=44,
        seed_model=45,
        patience=5,
        min_delta=0.01,
    )
    model = build_model(config)

    dummy_input = torch.randn(16, 3, 128, 128)
    output = model(dummy_input)

    assert output.shape == (16, 2)

    for name, param in model.named_parameters():
        if name.startswith(("conv1", "bn1", "layer1", "layer2", "layer3")):
            assert param.requires_grad is False
        else:
            assert param.requires_grad is True


def test_preprocessing_deterministic():
    transform = get_preprocessing_transforms(image_size=128)
    img_array = np.random.randint(0, 255, (300, 300, 3), dtype=np.uint8)
    img = Image.fromarray(img_array)

    tensor1 = transform(img)
    tensor2 = transform(img)

    assert torch.equal(tensor1, tensor2)
