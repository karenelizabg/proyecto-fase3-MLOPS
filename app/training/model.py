import json

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18
from training.config import TrainingConfig


def build_model(config: TrainingConfig) -> nn.Module:
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    
    for param in model.parameters():
        param.requires_grad = False
        
    num_ftrs = model.fc.in_features
    
    if config.hidden_layers > 0:
        model.fc = nn.Sequential(
            nn.Linear(num_ftrs, config.hidden_layers),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_layers, 2)
        )
    else:
        model.fc = nn.Sequential(
            nn.Dropout(config.dropout),
            nn.Linear(num_ftrs, 2)
        )
        
    return model

def export_architecture_and_class_map(
    arch_path: str = "architecture.json", 
    class_map_path: str = "class_map.json"
):
    with open(arch_path, "w") as f:
        json.dump(
            {
                "base_model": "resnet18",
                "pretrained": "IMAGENET1K_V1",
                "head": "configurable"
            },
            f
        )
        
    with open(class_map_path, "w") as f:
        json.dump({"3": "dog", "4": "cat"}, f)