import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from torchvision import datasets
from sklearn.model_selection import train_test_split

def create_datasets(data_dir, transforms):
    """
    Create and split datasets into train, validation, and test sets
    
    Args:
        data_dir (str): Directory containing the dataset
        transforms (dict): Dictionary with 'train' and 'val' transforms
        
    Returns:
        tuple: (train_dataset, val_dataset, test_dataset, class_names)
    """
    # Load the full dataset with training transforms initially
    full_dataset = datasets.ImageFolder(data_dir, transform=transforms['train'])
    
    # Get class names
    class_names = full_dataset.classes
    print(f"Found {len(class_names)} classes")
    
    # Get all indices and labels for stratification
    indices = list(range(len(full_dataset)))
    labels = [full_dataset[i][1] for i in indices]
    
    # Create stratified train/temp split (70/30)
    train_indices, temp_indices = train_test_split(
        indices, test_size=0.3, stratify=labels, random_state=42
    ) # we are doing a stratified split here in order to maintain the class distribution since we have imbalanced sets
    
    # Split temp into validation and test (each 15% of total)
    val_indices, test_indices = train_test_split(
        temp_indices, test_size=0.5, 
        stratify=[labels[i] for i in temp_indices], random_state=42
    ) # we are doing a stratified split here as well 
    
    # Create subset datasets with appropriate transforms
    train_dataset = Subset(full_dataset, train_indices)
    
    # For validation and test, we need to create a new dataset with val transforms
    val_full_dataset = datasets.ImageFolder(data_dir, transform=transforms['val'])
    val_dataset = Subset(val_full_dataset, val_indices)
    test_dataset = Subset(val_full_dataset, test_indices)
    
    print(f"Dataset splits: {len(train_dataset)} training, "
          f"{len(val_dataset)} validation, {len(test_dataset)} testing")
          
    return train_dataset, val_dataset, test_dataset, class_names