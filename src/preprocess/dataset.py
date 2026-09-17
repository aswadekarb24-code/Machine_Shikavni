import os
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms

# 1. Configuration
IMG_SIZE = 32  # Resize all images to 32x32
BATCH_SIZE = 64

# 2. Modern v2 Transforms (Scales [0, 255] PIL -> [-1.0, 1.0] Tensor)
transform_pipeline = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToImage(),
    transforms.ToDtype(torch.float32, scale=True),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# 3. Define dataset paths
ctsd_train_dir = "data/processed_data/CTSD/train/"
ctsd_test_dir = "data/processed_data/CTSD/test/"

gtsrb_train_dir = "data/processed_data/GTSRB/train/"
gtsrb_test_dir = "data/processed_data/GTSRB/test/"

btsd_train_dir = "data/processed_data/BTSD/train/"
btsd_test_dir = "data/processed_data/BTSD/test/"

# Helper function to fix string-sorting issue in ImageFolder
def fix_numeric_labels(dataset):
    """Re-maps targets to match the actual integer value of the folder names."""
    dataset.samples = [(path, int(os.path.basename(os.path.dirname(path)))) for path, _ in dataset.samples]
    dataset.targets = [target for _, target in dataset.samples]
    dataset.class_to_idx = {str(i): i for i in range(len(dataset.classes))}

# 4. Create Datasets using ImageFolder
ctsd_train_dataset = datasets.ImageFolder(root=ctsd_train_dir, transform=transform_pipeline)
fix_numeric_labels(ctsd_train_dataset)

ctsd_test_dataset = datasets.ImageFolder(root=ctsd_test_dir, transform=transform_pipeline)
fix_numeric_labels(ctsd_test_dataset)

gtsrb_train_dataset = datasets.ImageFolder(root=gtsrb_train_dir, transform=transform_pipeline)
fix_numeric_labels(gtsrb_train_dataset)

gtsrb_test_dataset = datasets.ImageFolder(root=gtsrb_test_dir, transform=transform_pipeline)
fix_numeric_labels(gtsrb_test_dataset)

btsd_train_dataset = datasets.ImageFolder(root=btsd_train_dir, transform=transform_pipeline)
fix_numeric_labels(btsd_train_dataset)

btsd_test_dataset = datasets.ImageFolder(root=btsd_test_dir, transform=transform_pipeline)
fix_numeric_labels(btsd_test_dataset)

# 5. Create DataLoaders
a_ctsd_train_loader = DataLoader(ctsd_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
a_ctsd_test_loader = DataLoader(ctsd_test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

a_gtsrb_train_loader = DataLoader(gtsrb_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
a_gtsrb_test_loader = DataLoader(gtsrb_test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

a_btsd_train_loader = DataLoader(btsd_train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
a_btsd_test_loader = DataLoader(btsd_test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# Sanity Check
print(f"CTSD Train Samples: {len(ctsd_train_dataset)}")
print(f"GTSRB Train Samples: {len(gtsrb_train_dataset)}")
print(f"BTSD Train Samples: {len(btsd_train_dataset)}")

# Grab one batch to verify shapes and integer class range
a, labels = next(iter(a_ctsd_train_loader))
print(f"Batch Image shape: {a.shape}")  # Expected: [64, 3, 32, 32]
print(f"Batch Label shape: {labels.shape}")  # Expected: [64]
print(f"Sample labels from batch: {labels[:10].tolist()}")