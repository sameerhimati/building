from torchvision import transforms
import torch
from torch.utils.data import DataLoader
from torchvision import datasets

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