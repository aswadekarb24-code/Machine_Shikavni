import torch
from PIL import Image
from src.noises.corruptions import CORRUPTION_REGISTRY
from src.noises.noise_loader import get_corrupted_loader
from src.models.cnn import TrafficSignCNN  # Adjust import based on your CNN class name

# 1. Sanity check: Direct corruption on a synthetic image
print("--- Step 1: Testing direct image corruption ---")
dummy_img = Image.new('RGB', (32, 32), color=(128, 64, 200))
corrupted_img = CORRUPTION_REGISTRY['gaussian_noise'](dummy_img, severity=3)
print(f"Corrupted Image Size: {corrupted_img.size}, Mode: {corrupted_img.mode}")

# 2. Sanity check: Test corrupted DataLoader
print("\n--- Step 2: Testing Corrupted DataLoader ---")
loader = get_corrupted_loader(dataset_name='GTSRB', corruption_name='gaussian_noise', severity=3)
a, labels = next(iter(loader))  # a holds the image tensor batch
print(f"Batch Tensor Shape: {a.shape}")    # Expected: torch.Size([64, 3, 32, 32])
print(f"Batch Labels Shape: {labels.shape}")  # Expected: torch.Size([64])

# 3. Sanity check: Forward batch through CNN model
print("\n--- Step 3: Testing Forward Pass through CNN ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TrafficSignCNN().to(device)
model.eval()

a = a.to(device)
with torch.no_grad():
    outputs = model(a)
    print(f"Model Output Logits Shape: {outputs.shape}")  # Expected: [64, num_classes]

print("\nAll pipeline checks passed successfully!")