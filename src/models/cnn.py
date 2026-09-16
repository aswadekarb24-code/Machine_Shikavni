import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms.v2 as transforms
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix
import matplotlib.pyplot as plt

from preprocess.dataset import a_gtsrb_train_loader, a_gtsrb_test_loader

# 1. Device Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 2. Modular CNN Architecture
class TrafficSignCNN(nn.Module):
    def __init__(self, in_channels=3, n_classes=43, img_size=32):
        super().__init__()
        
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 25, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # 32x32 -> 16x16
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(25, 50, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2, stride=2)  # 16x16 -> 8x8
        )
        
        self.block3 = nn.Sequential(
            nn.Conv2d(50, 75, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(75),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)  # 8x8 -> 4x4
        )
        
        # Calculate flattened feature dimensions dynamically
        flattened_size = 75 * (img_size // 8) * (img_size // 8)
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512, n_classes)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.classifier(x)

# 3. Training Function
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for x, y in dataloader:
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * x.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
        
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 4. Evaluation Function (Accuracy, F1-Score, Confusion Matrix)
def evaluate(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            loss = criterion(outputs, y)
            
            running_loss += loss.item() * x.size(0)
            preds = outputs.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            
    total = len(all_targets)
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    loss = running_loss / total
    acc = (all_preds == all_targets).mean()
    macro_f1 = f1_score(all_targets, all_preds, average='macro')
    cm = confusion_matrix(all_targets, all_preds)
    
    return loss, acc, macro_f1, cm

# Setup for GTSRB (Change n_classes to 58 for CTSD)
NUM_CLASSES = 43 
EPOCHS = 10

model = TrafficSignCNN(in_channels=3, n_classes=NUM_CLASSES, img_size=32).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_epoch(model, a_gtsrb_train_loader, optimizer, criterion)
    test_loss, test_acc, test_f1, cm = evaluate(model, a_gtsrb_test_loader, criterion)
    
    print(f"Epoch [{epoch}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
          f"Test Loss: {test_loss:.4f} Acc: {test_acc:.4f} F1: {test_f1:.4f}")

# Save the baseline model for Phase 3 noise evaluation
torch.save(model.state_dict(), "gtsrb_baseline_cnn.pth")