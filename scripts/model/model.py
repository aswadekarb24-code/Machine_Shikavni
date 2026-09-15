import torch
import torch.nn as nn

class DummyClassifier(nn.Module):
    """Placeholder model generating pseudo-random outputs for pipeline verification."""
    def __init__(self, num_classes: int = 43):
        super().__init__()
        self.num_classes = num_classes
        # Minimal layer so parameters exist for optimizer calls
        self.fc = nn.Linear(3 * 32 * 32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return torch.randn(batch_size, self.num_classes, device=x.device)