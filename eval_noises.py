import torch
from tqdm import tqdm
import pandas as pd
from src.models.cnn import TrafficSignCNN
from src.noises.corruptions import CORRUPTION_REGISTRY
from src.noises.noise_loader import get_corrupted_loader

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = ['CTSD', 'GTSRB', 'BTSD']
CORRUPTIONS = list(CORRUPTION_REGISTRY.keys())
SEVERITIES = [1, 2, 3, 4, 5]

def evaluate_model(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for a, labels in loader:
            a, labels = a.to(DEVICE), labels.to(DEVICE)
            outputs = model(a)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total if total > 0 else 0.0

# Load model and weights
model = TrafficSignCNN().to(DEVICE)
# model.load_state_dict(torch.load("checkpoints/model.pth"))

results = []
for dataset_name in DATASETS:
    for corruption in CORRUPTIONS:
        for severity in SEVERITIES:
            loader = get_corrupted_loader(dataset_name, corruption, severity)
            acc = evaluate_model(model, loader)
            results.append({
                'Dataset': dataset_name,
                'Corruption': corruption,
                'Severity': severity,
                'Accuracy': acc
            })
            print(f"[{dataset_name}] {corruption} (Severity {severity}): Accuracy = {acc:.4f}")

# Save evaluation benchmark
df = pd.DataFrame(results)
df.to_csv("noise_evaluation_results.csv", index=False)