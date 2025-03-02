import torch
import torch.nn as nn
from torchvision import models

def create_efficientnet_model(num_classes, pretrained=True, freeze_backbone=True):
    """
    Create an EfficientNetV2-S model with optional pretraining and backbone freezing
    
    Args:
        num_classes (int): Number of output classes
        pretrained (bool): Whether to use ImageNet pretrained weights
        freeze_backbone (bool): Whether to freeze the backbone layers
        
    Returns:
        model: PyTorch model
    """
    # Load pre-trained EfficientNetV2-S
    if pretrained:
        weights = 'IMAGENET1K_V1'
    else:
        weights = None
        
    model = models.efficientnet_v2_s(weights=weights)
    
    # Freeze the backbone if requested
    if freeze_backbone and pretrained:
        for param in model.parameters():
            param.requires_grad = False
    
    # Replace the classifier
    num_features = model.classifier[1].in_features  # Takes the number of features in the last layer (input dimension)
    model.classifier[1] = nn.Linear(num_features, num_classes) # Replace the last layer with same number of input dimension and output dimension of our classes
    
    # Print trainable parameters info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    return model