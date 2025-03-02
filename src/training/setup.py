import torch
import torch.nn as nn
import torch.optim as optim

def setup_training(model, device, learning_rate=0.001):
    """
    Set up training components: loss function, optimizer, and scheduler
    
    Args:
        model: PyTorch model
        device: Device to train on (cuda, mps, or cpu)
        learning_rate: Initial learning rate
        
    Returns:
        tuple: (criterion, optimizer, scheduler)
    """
    # Move model to device
    model = model.to(device)
    
    # Cross Entropy Loss for multi-class classification
    criterion = nn.CrossEntropyLoss()
    
    # Use Adam optimizer for the classifier parameters only
    # Only optimize parameters that require gradients
    optimizer = optim.Adam(
        [p for p in model.parameters() if p.requires_grad],
        lr=learning_rate
    )
    
    # Learning rate scheduler that reduces LR when validation loss plateaus
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min',           # Minimize validation loss
        factor=0.1,           # Reduce LR by 90% when triggered
        patience=3,           # Number of epochs with no improvement after which LR will be reduced
        verbose=True          # Print message when LR is reduced
    )
    
    return criterion, optimizer, scheduler