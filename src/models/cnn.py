from torch import nn

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
