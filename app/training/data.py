import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def get_transforms(split: str, image_size: int = 224):
    if split == "train":
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        return transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )


class ManifestDataset(Dataset):
    def __init__(self, manifest_path: str, split: str, image_size: int = 224):
        df = pd.read_csv(manifest_path)
        self.data = df[df["split"] == split].reset_index(drop=True)
        self.transform = get_transforms(split, image_size)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image = Image.open(row["path"]).convert("RGB")
        tensor = self.transform(image)
        label = 0 if row["category_id"] == 3 else 1
        return tensor, label, row["crop_id"]


def create_dataloader(manifest_path: str, split: str, batch_size: int, seed: int):
    dataset = ManifestDataset(manifest_path, split)
    generator = torch.Generator()
    generator.manual_seed(seed)

    return DataLoader(
        dataset, batch_size=batch_size, shuffle=(split == "train"), generator=generator
    )
