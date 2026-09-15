from pathlib import Path
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score

def train_and_evaluate(
    model: torch.nn.Module,
    test_loader,
    results_path: Path,
    device: str = "cpu"
) -> pd.DataFrame:
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    res_df = pd.DataFrame([{
        "experiment": "baseline_placeholder",
        "accuracy": round(acc, 4),
        "f1_score": round(f1, 4)
    }])

    results_path.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(results_path, index=False)
    return res_df