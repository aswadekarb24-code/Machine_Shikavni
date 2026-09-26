import torch.nn as nn
from torchvision import models
from src.models.cnn import TrialCNN


def get_model(model_name: str, in_channels: int = 3, n_classes: int = 43, pretrained: bool = False):
    name = model_name.lower()
    
    if name == 'trialcnn':
        return TrialCNN(in_channels=in_channels, n_classes=n_classes)

    elif name == 'resnet18':
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        
        if in_channels != 3:
            model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, n_classes)
        return model

    elif name == 'mobilenet_v2':
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = models.mobilenet_v2(weights=weights)
        
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
        return model

    elif name == 'efficientnet_b0':
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        
        if in_channels != 3:
            model.features[0][0] = nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1, bias=False)
            
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, n_classes)
        return model

    else:
        raise ValueError(f"Unknown model architecture: {model_name}")