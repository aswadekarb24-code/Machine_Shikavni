from torch import nn, flatten
from torch.nn import functional as F

from src.config import IMG_SIZE


class TrialCNN(nn.Module):
    def __init__(self, in_channels=3,n_classes=43, img_size = IMG_SIZE) -> None:
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels,25,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(25),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(25,50,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(50),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.MaxPool2d(2,stride=2)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(50,75,kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(75),
            nn.ReLU(),
            nn.MaxPool2d(2,stride=2)
        )

        flat_sz = 75 * (img_size//8) *(img_size//8)
        self.fin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_sz, 512),
            nn.Dropout(0.3),
            nn.ReLU(),
            nn.Linear(512,n_classes)
        )

    def forward(self,x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        return self.fin(x)

class TSRDClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=58):
        super(TSRDClassifier, self).__init__()
        
        # 1. Conv2D + ReLU (64 filters, 5x5)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=5, padding=2)
        
        # 2. Conv2D + ReLU (64 filters, 5x5)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=5, padding=2)
        
        # 3. MaxPooling2D (2x2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 4. Conv2D (64 filters, 3x3) - No activation specified in table
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 5. Conv2D + ReLU (64 filters, 3x3)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        
        # 6. MaxPooling2D (2x2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 7. Flatten + 8. Dense + ReLU (128 units)
        # Assuming standard 32x32 input images (e.g., GTSRB):
        # After two max pooling layers, spatial dims go 32 -> 16 -> 8.
        # Flatten size = 64 channels * 8 * 8 = 4096
        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        
        # 9. Dropout (0.1)
        self.dropout = nn.Dropout(p=0.1)
        
        # 10. Dense / Output layer (58 units for 58 classes)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Layer 1 & 2: Conv2D + ReLU
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        
        # Layer 3: MaxPooling2D
        x = self.pool1(x)
        
        # Layer 4: Conv2D (linear activation)
        x = self.conv3(x)
        
        # Layer 5: Conv2D + ReLU
        x = F.relu(self.conv4(x))
        
        # Layer 6: MaxPooling2D
        x = self.pool2(x)
        
        # Layer 7: Flatten
        x = flatten(x, start_dim=1)
        
        # Layer 8: Dense + ReLU
        x = F.relu(self.fc1(x))
        
        # Layer 9: Dropout
        x = self.dropout(x)
        
        # Layer 10: Dense (Logits)
        x = self.fc2(x)
        
        return x