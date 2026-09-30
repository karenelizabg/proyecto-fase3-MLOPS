import pytest


@pytest.fixture
def mock_dataset_env(tmp_path):
    import numpy as np
    import pandas as pd
    from PIL import Image

    base_dir = tmp_path / "data/derived/crops"
    base_dir.mkdir(parents=True)

    records = []
    for i in range(4):
        img_name = f"dummy_{i}.jpg"
        img_array = np.random.randint(0, 255, (10, 10, 3), dtype=np.uint8)
        Image.fromarray(img_array).save(base_dir / img_name)

        if i < 3:
            records.append({"path": img_name, "split": "train", "label": 0, "crop_id": i})
        else:
            records.append({"path": img_name, "split": "val", "label": 1, "crop_id": i})
            records.append({"path": img_name, "split": "test", "label": 1, "crop_id": i + 1})

    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(records).to_csv(manifest_path, index=False)

    return str(manifest_path), str(base_dir)
