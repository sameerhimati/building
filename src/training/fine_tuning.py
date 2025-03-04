import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import copy
import time
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

def freeze_model_backbone(model, freeze=True):
    """
    Freeze or unfreeze all backbone layers in an EfficientNet model
    
    Args:
        model: The EfficientNet model
        freeze: Whether to freeze (True) or unfreeze (False) the backbone
        
    Returns:
        model: The model with updated frozen/unfrozen layers
    """
    # Freeze/unfreeze backbone
    for param in model.features.parameters():
        param.requires_grad = not freeze
        
    return model

def unfreeze_layers_by_index(model, indices_to_unfreeze):
    """
    Unfreeze specific blocks of an EfficientNet model by their indices
    
    Args:
        model: The EfficientNet model
        indices_to_unfreeze: List of block indices to unfreeze (0-6 for EfficientNetV2)
        
    Returns:
        model: The updated model
    """
    # First freeze all backbone layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Then unfreeze specified blocks
    for block_idx in indices_to_unfreeze:
        if 0 <= block_idx < len(model.features):
            for param in model.features[block_idx].parameters():
                param.requires_grad = True
    
    return model

def get_parameter_stats(model):
    """
    Get statistics about trainable parameters in the model
    
    Args:
        model: PyTorch model
        
    Returns:
        dict: Statistics about trainable parameters
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Get per-layer parameter counts
    layer_stats = {}
    for name, module in model.named_modules():
        if hasattr(module, 'weight') and hasattr(module.weight, 'shape'):
            params = module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None:
                params += module.bias.numel()
            
            trainable = 0
            if hasattr(module, 'weight') and module.weight.requires_grad:
                trainable += module.weight.numel()
            if hasattr(module, 'bias') and module.bias is not None and module.bias.requires_grad:
                trainable += module.bias.numel()
                
            layer_stats[name] = {
                'params': params,
                'trainable': trainable,
                'trainable_pct': trainable / params if params > 0 else 0
            }
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'trainable_pct': trainable_params / total_params,
        'layers': layer_stats
    }

def setup_discriminative_learning_rates(model, base_lr=1e-4, backbone_multiplier=0.1):
    """
    Set up discriminative learning rates - lower rates for backbone, higher for classifier
    
    Args:
        model: PyTorch model
        base_lr: Base learning rate for the classifier head
        backbone_multiplier: Multiplier to reduce learning rate for backbone layers
        
    Returns:
        optimizer: Configured optimizer with different learning rates
    """
    # Separate parameters into backbone and classifier
    backbone_params = []
    classifier_params = []
    
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'features' in name:
                backbone_params.append(param)
            else:
                classifier_params.append(param)
    
    # Create parameter groups with different learning rates
    param_groups = [
        {'params': backbone_params, 'lr': base_lr * backbone_multiplier},
        {'params': classifier_params, 'lr': base_lr}
    ]
    
    return Adam(param_groups)

def fine_tune_model(model, dataloaders, device, criterion=None, 
                    unfreeze_schedule=None, num_epochs=10,
                    base_lr=1e-4, backbone_lr_multiplier=0.1,
                    patience=3, factor=0.5, min_lr=1e-6):
    """
    Fine-tune a model with gradual unfreezing
    
    Args:
        model: The model to fine-tune
        dataloaders: Dictionary with 'train' and 'val' data loaders
        device: Device to train on
        criterion: Loss function (defaults to CrossEntropyLoss if None)
        unfreeze_schedule: Dictionary mapping epoch to list of block indices to unfreeze
                           e.g., {0: [], 3: [6], 5: [5, 6]}
        num_epochs: Total number of training epochs
        base_lr: Base learning rate for classifier head
        backbone_lr_multiplier: Multiplier for backbone learning rate
        patience: Patience for learning rate reduction
        factor: Factor for learning rate reduction
        min_lr: Minimum learning rate
        
    Returns:
        model: Fine-tuned model
        history: Training history dictionary
    """
    # Set up loss function if not provided
    if criterion is None:
        criterion = nn.CrossEntropyLoss()
    
    # Default unfreeze schedule if not provided (gradually unfreeze from top to bottom)
    if unfreeze_schedule is None:
        # For a 10-epoch schedule
        unfreeze_schedule = {
            0: [],            # Start with all frozen except classifier
            int(num_epochs * 0.3): [6],  # Unfreeze last block at 30% of training
            int(num_epochs * 0.5): [5, 6],  # Unfreeze blocks 5-6 at 50% of training
            int(num_epochs * 0.7): [4, 5, 6]  # Unfreeze blocks 4-6 at 70% of training
        }
    
    # Initialize history dictionary to track metrics
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'param_counts': []
    }
    
    # Initialize best model and performance tracking
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # Start timing
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Check if we need to update unfreezing according to schedule
        if epoch in unfreeze_schedule:
            blocks_to_unfreeze = unfreeze_schedule[epoch]
            model = unfreeze_layers_by_index(model, blocks_to_unfreeze)
            
            # Get parameter statistics after unfreezing
            param_stats = get_parameter_stats(model)
            history['param_counts'].append(param_stats)
            
            print(f"Unfreezing blocks {blocks_to_unfreeze}")
            print(f"Trainable parameters: {param_stats['trainable']:,}/{param_stats['total']:,} "
                  f"({param_stats['trainable_pct']:.2%})")
        
        # Set up optimizer with discriminative learning rates
        optimizer = setup_discriminative_learning_rates(
            model, base_lr=base_lr, backbone_multiplier=backbone_lr_multiplier
        )
        
        # Set up learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='max', factor=factor, 
            patience=patience, verbose=True, min_lr=min_lr
        )
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode
            
            running_loss = 0.0
            running_corrects = 0
            total_samples = 0

            # Create progress bar for this phase
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()}')
            
            # Iterate over data batches
            for inputs, labels in pbar:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass - track history only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    # Backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                # Statistics
                batch_size = inputs.size(0)
                batch_loss = loss.item() * batch_size
                batch_corrects = torch.sum(preds == labels.data).item()
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data).item()
                total_samples += batch_size

                batch_acc = batch_corrects / batch_size
                pbar.set_postfix({
                    'loss': f'{batch_loss/batch_size:.4f}',
                    'acc': f'{batch_acc:.4f}'
                })
            
            epoch_loss = running_loss / total_samples
            epoch_acc = running_corrects / total_samples
            
            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            # Track history
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)
                
                # Update learning rate based on validation accuracy
                scheduler.step(epoch_acc)
                
                # Deep copy the model if it's the best so far
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f'New best validation accuracy: {best_acc:.4f}')
        
        print()  # Empty line between epochs
    
    # Calculate total training time
    time_elapsed = time.time() - start_time
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best validation accuracy: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    return model, history

def plot_fine_tuning_history(history, save_path=None):
    """
    Plot training and validation metrics from fine-tuning
    
    Args:
        history: Training history dictionary
        save_path: Optional path to save the plot
        
    Returns:
        None
    """
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 10))
    
    # Plot accuracy
    ax1.plot(history['train_acc'], 'b-', label='Training')
    ax1.plot(history['val_acc'], 'r-', label='Validation')
    ax1.set_title('Model Accuracy During Fine-tuning')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()
    ax1.grid(True)
    
    # Plot loss
    ax2.plot(history['train_loss'], 'b-', label='Training')
    ax2.plot(history['val_loss'], 'r-', label='Validation')
    ax2.set_title('Model Loss During Fine-tuning')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved to {save_path}")
    
    plt.show()
    
def plot_parameter_unfreezing(history, save_path=None):
    """
    Plot the progression of trainable parameters during fine-tuning
    
    Args:
        history: Training history dictionary with param_counts
        save_path: Optional path to save the plot
        
    Returns:
        None
    """
    if 'param_counts' not in history or not history['param_counts']:
        print("No parameter count history available")
        return
    
    # Extract data from history
    epochs = []
    trainable_counts = []
    trainable_pcts = []
    
    for i, param_stat in enumerate(history['param_counts']):
        epochs.append(i)
        trainable_counts.append(param_stat['trainable'])
        trainable_pcts.append(param_stat['trainable_pct'] * 100)  # Convert to percentage
    
    # Create figure with multiple subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot trainable parameter count
    ax1.plot(epochs, trainable_counts, 'go-')
    ax1.set_title('Trainable Parameters During Fine-tuning')
    ax1.set_ylabel('Number of Parameters')
    ax1.set_xlabel('Epoch')
    ax1.grid(True)
    
    # Plot trainable parameter percentage
    ax2.plot(epochs, trainable_pcts, 'mo-')
    ax2.set_title('Percentage of Trainable Parameters')
    ax2.set_ylabel('Percentage (%)')
    ax2.set_xlabel('Epoch')
    ax2.grid(True)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save_path:
        plt.savefig(save_path)
        print(f"Parameter unfreezing plot saved to {save_path}")
    
    plt.show()