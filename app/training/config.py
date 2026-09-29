import json

from pydantic import BaseModel, ConfigDict, Field


class TrainingConfig(BaseModel):
    model_config = ConfigDict(strict=True)

    optimizer: str = Field(..., pattern="^(Adam|SGD)$")
    batch_size: int = Field(..., gt=0)
    max_epochs: int = Field(..., gt=0)
    learning_rate: float = Field(..., gt=0.0)
    image_size: int = Field(..., ge=32)
    hidden_layers: int = Field(..., ge=0)
    dropout: float = Field(..., ge=0.0, le=1.0)
    
    seed_split: int
    seed_train: int
    seed_model: int
    seed_eval: int
    
    patience: int = Field(..., ge=0)
    min_delta: float = Field(..., ge=0.0)

def export_schema(filepath: str = "training_schema.json"):
    schema = TrainingConfig.model_json_schema()
    with open(filepath, "w") as f:
        json.dump(schema, f, indent=2)