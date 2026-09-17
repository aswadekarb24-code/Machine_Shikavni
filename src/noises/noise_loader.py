import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# Import base datasets and the transform pipeline from your existing dataset setup
from src.preprocess.dataset import (
    ctsd_test_dataset, 
    gtsrb_test_dataset, 
    btsd_test_dataset,
    transform_pipeline,
    BATCH_SIZE
)
from src.noises.corruptions import CORRUPTION_REGISTRY

class CorruptedDataset(Dataset):
    """Wraps an existing dataset to apply specific corruptions on the fly."""
    
    def __init__(self, base_dataset, corruption_name, severity, transform=None):
        self.base_dataset = base_dataset
        self.corruption_name = corruption_name
        self.severity = severity
        self.transform = transform
        self.corruption_func = CORRUPTION_REGISTRY.get(corruption_name)
        
        if not self.corruption_func:
            raise ValueError(f"Corruption '{corruption_name}' not found in registry.")

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Retrieve the raw image path and label from the base dataset's underlying samples
        img_path, label = self.base_dataset.samples[idx]
        
        # Open raw image
        img = Image.open(img_path).convert("RGB")
        
        # 1. Apply the specific noise corruption
        corrupted_img = self.corruption_func(img, self.severity)
        
        # 2. Apply standard preprocessing (Resize, ToTensor, Normalize)
        if self.transform:
            corrupted_tensor = self.transform(corrupted_img)
        else:
            corrupted_tensor = corrupted_img
            
        return corrupted_tensor, label

def get_corrupted_loader(dataset_name, corruption_name, severity=1):
    """Factory function to return a DataLoader for a specific corrupted dataset."""
    
    datasets_map = {
        'CTSD': ctsd_test_dataset,
        'GTSRB': gtsrb_test_dataset,
        'BTSD': btsd_test_dataset
    }
    
    if dataset_name not in datasets_map:
        raise ValueError(f"Dataset {dataset_name} not supported.")
        
    base_data = datasets_map[dataset_name]
    
    # Wrap in our corruption class
    corrupted_data = CorruptedDataset(
        base_dataset=base_data,
        corruption_name=corruption_name,
        severity=severity,
        transform=transform_pipeline
    )
    
    return DataLoader(
        corrupted_data, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        pin_memory=True
    )