import torch
import torch.nn as nn
from torchvision import models


def get_model(model_name='efficientnet', num_classes=2, pretrained=True):
    if model_name == 'efficientnet':
        model = models.efficientnet_b0(
            weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    elif model_name == 'mobilenet':
        model = models.mobilenet_v3_small(
            weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)
    elif model_name == 'resnet18':
        model = models.resnet18(
            weights='IMAGENET1K_V1' if pretrained else None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(in_features, num_classes)
        )
    return model