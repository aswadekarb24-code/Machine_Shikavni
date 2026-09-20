from torch.utils.data import DataLoader
from src.preprocess.dataset import (
    gtsrb_train_dataset, gtsrb_test_dataset,
    btsd_train_dataset, btsd_test_dataset,
    ctsd_train_dataset, ctsd_test_dataset,
    transform_pipeline, BATCH_SIZE
)
from src.preprocess.dynamic_dataset import DynamicCorruptedDataset

DATASET_CONFIGS = {
    'GTSRB': {'train': gtsrb_train_dataset, 'test': gtsrb_test_dataset, 'n_classes': 43},
    'BTSD':  {'train': btsd_train_dataset,  'test': btsd_test_dataset,  'n_classes': 62},
    'CTSD':  {'train': ctsd_train_dataset,  'test': ctsd_test_dataset,  'n_classes': 58}
}

def get_dataset_loaders(dataset_name: str, batch_size: int = BATCH_SIZE, use_dynamic_noise: bool = False, noise_prob: float = 0.5):
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Choose from {list(DATASET_CONFIGS.keys())}")

    config = DATASET_CONFIGS[dataset_name]
    raw_train_ds = config['train']
    raw_test_ds = config['test']

    # Select clean or dynamically corrupted training dataset
    if use_dynamic_noise:
        train_ds = DynamicCorruptedDataset(raw_train_ds, corruption_prob=noise_prob, transform=transform_pipeline)
    else:
        train_ds = raw_train_ds

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        raw_test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=4, pin_memory=True
    )

    return train_loader, test_loader, config['n_classes']