import random
from PIL import Image
from torch.utils.data import Dataset
from src.noises.corruptions import CORRUPTION_REGISTRY

class DynamicCorruptedDataset(Dataset):
    """
    Wraps an ImageFolder dataset to dynamically apply random noise corruptions
    with probability `corruption_prob` on each data fetch call.
    """
    def __init__(self, base_dataset, corruption_prob=0.5, transform=None):
        self.base_dataset = base_dataset
        self.corruption_prob = corruption_prob
        self.transform = transform
        self.corruption_keys = list(CORRUPTION_REGISTRY.keys())

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        img_path, label = self.base_dataset.samples[idx]
        img = Image.open(img_path).convert("RGB")

        # Inject noise dynamically with probability p
        if random.random() < self.corruption_prob:
            chosen_noise = random.choice(self.corruption_keys)
            severity = random.randint(1, 5)
            img = CORRUPTION_REGISTRY[chosen_noise](img, severity=severity)

        if self.transform:
            img = self.transform(img)

        return img, label