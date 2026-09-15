from pathlib import Path
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class GTSRBDataset(Dataset):
    def __init__(self, csv_file: Path, root_dir: Path, transform=None):
        self.data = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        row = self.data.iloc[idx]
        img_path = self.root_dir / row["Path"]
        image = Image.open(img_path).convert("RGB")
        label = int(row["ClassId"])

        if self.transform:
            image = self.transform(image)

        return image, label

def create_dataloaders(
    data_dir: Path,
    batch_size: int = 64,
    transform=None
) -> tuple[DataLoader, DataLoader]:
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    train_ds = GTSRBDataset(data_dir / "Train.csv", data_dir, transform=transform)
    test_ds = GTSRBDataset(data_dir / "Test.csv", data_dir, transform=transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader