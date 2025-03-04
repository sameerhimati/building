from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
import numpy as np

def calculate_dataset_stats(data_dir):
    """
    Calculate the mean and standard deviation of your dataset.
    
    Args:
        data_dir (str): Path to the dataset directory
    
    Returns:
        tuple: (mean, std) for each channel
    """
    # First, create a dataset with just ToTensor (no normalization yet)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    
    # Load the dataset
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=64, num_workers=4, shuffle=False)
    
    # Initialize variables
    channels_sum = torch.zeros(3)
    channels_squared_sum = torch.zeros(3)
    num_batches = 0
    
    # Calculate sum and squared sum for each channel
    for images, _ in dataloader:
        channels_sum += torch.mean(images, dim=[0, 2, 3]) # images shape = [batch_size = 64, channels = 3, width = 224, height = 224] avg across 0, 2, 3 gives us a tensor of shape [3]
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3]) # i.e. the shape is [3 numbers, one for each channel, its avg and squared avg]
        num_batches += 1
    
    # Calculate mean and std
    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean**2)**0.5
    
    return mean.tolist(), std.tolist()

def get_data_transforms(data_dir=None, use_custom_stats=False):
    """
    Creates transformation pipelines for training and validation/testing.
    
    Returns:
        dict: Dictionary containing 'train' and 'val' transformation pipelines
    """

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Calculate custom statistics if requested
    if data_dir and use_custom_stats:
        print("Calculating dataset statistics...")
        try:
            mean, std = calculate_dataset_stats(data_dir)
            print(f"Custom dataset statistics - Mean: {mean}, Std: {std}")
        except Exception as e:
            print(f"Error calculating statistics: {e}")
            print("Falling back to ImageNet statistics")
    else:
        print("Using ImageNet statistics for normalization")


    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256), # Makes images slightly larger than needed
            transforms.RandomCrop(224), # Crops to required size while adding positional variation
            transforms.RandomHorizontalFlip(), # Architectural style is usually preserved when flipped
            transforms.RandomRotation(10), # Small rotations simulate different camera angles
            transforms.ColorJitter(brightness=0.2, contrast=0.2), # Helps model be robust to lighting conditions
            transforms.ToTensor(), # Converts PIL images to PyTorch tensors
            transforms.Normalize(mean, std) # The values are the mean and standard deviation of each color channel (RGB) calculated from the entire ImageNet dataset.
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    }
    return data_transforms

def get_prediction_transform(img_size=224, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Returns transformation pipeline for prediction, with robust image handling
    """
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    return transform

def preprocess_image_for_prediction(image, transform=None):
    """
    Robustly preprocess any image for model prediction
    
    Args:
        image: Input image (PIL Image, numpy array, or tensor)
        transform: Optional custom transform (uses default if None)
        
    Returns:
        torch.Tensor: Preprocessed image tensor ready for model input (with batch dimension)
    """
    # If no transform provided, use default
    if transform is None:
        transform = get_prediction_transform()
    
    # Convert to PIL Image if it's not already
    if not isinstance(image, Image.Image):
        if isinstance(image, torch.Tensor):
            # If it's a tensor, convert to numpy first
            image = Image.fromarray(image.numpy().astype(np.uint8))
        else:
            # Assume it's a numpy array or something convertible to numpy
            image = Image.fromarray(np.array(image))
    
    # Ensure image is in RGB mode (convert from RGBA if needed)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transformations and add batch dimension
    transformed_image = transform(image)
    input_tensor = transformed_image.unsqueeze(0)
    
    return input_tensor