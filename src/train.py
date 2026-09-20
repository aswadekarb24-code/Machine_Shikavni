import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

from src.models.cnn import TrafficSignCNN
from src.preprocess.dataset_factory import get_dataset_loaders

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_one_epoch(model, dataloader, optimizer, criterion, scaler):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for x, y in dataloader:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        optimizer.zero_grad()

        # Mixed Precision forward pass for faster VRAM utilization
        with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
            outputs = model(x)
            loss = criterion(outputs, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / total, correct / total

def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss, all_preds, all_targets = 0.0, [], []

    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=(DEVICE.type == 'cuda')):
                outputs = model(x)
                loss = criterion(outputs, y)

            running_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

    total = len(all_targets)
    acc = (np.array(all_preds) == np.array(all_targets)).mean()
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    return running_loss / total, acc, macro_f1

def run_experiment(dataset_name: str, epochs: int = 50, use_dynamic_noise: bool = False, noise_prob: float = 0.5):
    print(f"\n================ Start Experiment: Dataset={dataset_name} | DynamicNoise={use_dynamic_noise} ================")
    
    train_loader, test_loader, n_classes = get_dataset_loaders(
        dataset_name, batch_size=128, use_dynamic_noise=use_dynamic_noise, noise_prob=noise_prob
    )

    model = TrafficSignCNN(in_channels=3, n_classes=n_classes, img_size=32).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda', enabled=(DEVICE.type == 'cuda'))

    best_f1 = 0.0
    history = []
    os.makedirs("checkpoints", exist_ok=True)
    
    suffix = "robust" if use_dynamic_noise else "baseline"
    checkpoint_path = f"checkpoints/{dataset_name.lower()}_{suffix}_model.pth"

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        test_loss, test_acc, test_f1 = evaluate(model, test_loader, criterion)
        scheduler.step()

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'test_loss': test_loss, 'test_acc': test_acc, 'test_f1': test_f1
        })
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:03d}/{epochs:03d}] | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f} | Test F1: {test_f1:.4f}")

        # Save model with highest Macro F1 score
        if test_f1 > best_f1:
            best_f1 = test_f1
            torch.save(model.state_dict(), checkpoint_path)
            print(f" Saved Best Checkpoint -> {checkpoint_path} (F1: {best_f1:.4f})")

    # Save training logs
    os.makedirs("results", exist_ok=True)
    pd.DataFrame(history).to_csv(f"results/{dataset_name.lower()}_{suffix}_history.csv", index=False)

if __name__ == "__main__":
    # 1. Train Clean Baselines for all datasets
    # for ds in ['GTSRB', 'BTSD', 'CTSD']:
    #     run_experiment(dataset_name=ds, epochs=50, use_dynamic_noise=False)

    # 2. Train Robust Models (with 50% random noise injection during training)
    for ds in ['GTSRB', 'BTSD', 'CTSD']:
        epochs = 10 if ds == 'GTSRB' else 200
        run_experiment(dataset_name=ds, epochs=epochs, use_dynamic_noise=True, noise_prob=0.5)