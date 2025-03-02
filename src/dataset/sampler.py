import torch
import numpy as np
from torch.utils.data import WeightedRandomSampler

def create_weighted_sampler(dataset):
    """
    Create a weighted sampler to handle class imbalance
    
    Args:
        dataset: PyTorch dataset with targets attribute or a Subset with dataset.dataset.targets
        
    Returns:
        WeightedRandomSampler: Sampler that over-samples minority classes
    """
    # Handle both regular datasets and Subset datasets
    if hasattr(dataset, 'targets'):
        # Regular dataset
        targets = dataset.targets
    elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'targets'):
        # Subset dataset - we need the indices to get the right targets
        targets = [dataset.dataset.targets[i] for i in dataset.indices]
    else:
        # Dataset doesn't have targets attribute - try to get labels manually
        # This works for ImageFolder with a Subset
        if hasattr(dataset, 'indices'):
            targets = [dataset.dataset.samples[i][1] for i in dataset.indices]
        else:
            targets = [sample[1] for sample in dataset.samples]
    
    # Count samples per class
    class_counts = np.bincount(targets)
    print(f"Class counts: Min={min(class_counts)}, Max={max(class_counts)}, "
          f"Imbalance ratio={max(class_counts)/min(class_counts):.2f}")
    
    # Calculate weights (inverse frequency)
    class_weights = 1.0 / class_counts
    
    # Normalize weights so they sum to 1
    class_weights = class_weights / class_weights.sum()
    
    # Assign weights to each sample
    sample_weights = [class_weights[t] for t in targets]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights, 
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler

def create_dataloaders(train_dataset, val_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    Create DataLoaders for train, validation and test datasets
    
    Args:
        train_dataset: Training dataset
        val_dataset: Validation dataset
        test_dataset: Test dataset
        batch_size: Batch size for training and evaluation
        num_workers: Number of workers for data loading
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Create weighted sampler for training data
    train_sampler = create_weighted_sampler(train_dataset)
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        sampler=train_sampler,  # Use the weighted sampler
        num_workers=num_workers,
        pin_memory=True  # Speeds up data transfer to GPU
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle validation data
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # No need to shuffle test data
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader