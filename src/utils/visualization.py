import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import OrderedDict

def visualize_model_structure(model, input_size=(3, 224, 224), save_path=None):
    """
    Visualize the structure of a PyTorch model, showing layer dimensions.
    
    Args:
        model: PyTorch model
        input_size: Input dimensions (channels, height, width)
        save_path: Optional path to save visualization
    """
    # For detailed layer-by-layer analysis
    print("\nDetailed Layer Analysis:")
    device = next(model.parameters()).device
    dummy_input = torch.zeros((1,) + input_size).to(device)
    
    # Register hook to capture layer outputs
    layer_dims = []
    hooks = []
    
    def hook_fn(name):
        def hook(module, input, output):
            layer_dims.append({
                'Layer': name,
                'Input Shape': str(tuple(input[0].shape)),
                'Output Shape': str(tuple(output.shape)),
                'Parameters': sum(p.numel() for p in module.parameters()),
                'Trainable': sum(p.numel() for p in module.parameters() if p.requires_grad)
            })
        return hook
    
    # Attach hooks to track dimensions
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear, 
                              torch.nn.BatchNorm2d, torch.nn.MaxPool2d)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    # Run a forward pass
    with torch.no_grad():
        model(dummy_input)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Display results as a table
    df = pd.DataFrame(layer_dims)
    print(df)
    
    # Calculate parameter statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params:.2%})")
    
    if save_path:
        # Save DataFrame
        df.to_csv(save_path + ".csv")
        
        # Create visualization of layer dimensions
        if len(layer_dims) > 0:
            plt.figure(figsize=(12, 8))
            
            # Extract layers with output dimensions for plotting
            layer_names = [item['Layer'].split('.')[-1][:10] for item in layer_dims]
            # Get output shapes and convert to tensor sizes
            output_shapes = [eval(item['Output Shape'])[1] for item in layer_dims]
            
            # Plot
            plt.bar(range(len(output_shapes)), output_shapes, color='skyblue')
            plt.xticks(range(len(layer_names)), layer_names, rotation=90)
            plt.ylabel('Feature Dimensions')
            plt.title('Feature Dimensions Across Network Layers')
            plt.tight_layout()
            plt.savefig(save_path + "_dims.png")
            
            # Create a plot showing trainable vs non-trainable parameters
            plt.figure(figsize=(10, 6))
            
            trainable = [item['Trainable'] for item in layer_dims]
            nontrainable = [item['Parameters'] - item['Trainable'] for item in layer_dims]
            
            # Only include layers with parameters
            param_layers = [i for i, item in enumerate(layer_dims) if item['Parameters'] > 0]
            param_layer_names = [layer_names[i] for i in param_layers]
            param_trainable = [trainable[i] for i in param_layers]
            param_nontrainable = [nontrainable[i] for i in param_layers]
            
            # Create stacked bar chart
            plt.figure(figsize=(12, 8))
            plt.bar(range(len(param_layers)), param_trainable, color='green', label='Trainable')
            plt.bar(range(len(param_layers)), param_nontrainable, bottom=param_trainable, 
                   color='red', label='Frozen')
            
            plt.xticks(range(len(param_layers)), param_layer_names, rotation=90)
            plt.yscale('log')
            plt.ylabel('Number of Parameters (log scale)')
            plt.title('Trainable vs. Frozen Parameters by Layer')
            plt.legend()
            plt.tight_layout()
            plt.savefig(save_path + "_params.png")
            
            print(f"Visualizations saved to {save_path}")
        
    return df

def visualize_feature_maps(model, image_tensor, layer_names=None, save_path=None):
    """
    Visualize feature maps from different layers of the model.
    
    Args:
        model: PyTorch model
        image_tensor: Input image tensor (1, C, H, W)
        layer_names: List of layer names to visualize (if None, selects some automatically)
        save_path: Optional path to save visualization
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)
    
    # Dictionary to store feature maps
    feature_maps = OrderedDict()
    
    # Function to get feature maps
    def get_feature_maps(name):
        def hook(model, input, output):
            feature_maps[name] = output.detach().cpu()
        return hook
    
    # If layer_names not specified, get some key layers automatically
    if layer_names is None:
        layer_names = []
        # For EfficientNet, get outputs from start, middle and end of features
        if hasattr(model, 'features'):
            if len(model.features) >= 7:  # EfficientNetV2 has 7 blocks
                layer_names.extend(['features.0', 'features.3', 'features.6'])
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if name in layer_names:
            hooks.append(module.register_forward_hook(get_feature_maps(name)))
    
    # Forward pass
    with torch.no_grad():
        model(image_tensor)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    # Visualize feature maps
    if len(feature_maps) > 0:
        for layer_name, feature_map in feature_maps.items():
            # Get the first 16 channels (or fewer if less available)
            num_channels = min(16, feature_map.shape[1])
            
            # Create grid for visualization
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f'Feature Maps: {layer_name}', fontsize=16)
            
            # Flatten axes for easy iteration
            axes = axes.flatten()
            
            for i in range(num_channels):
                # Get feature map for channel i
                channel_map = feature_map[0, i].numpy()
                
                # Plot
                axes[i].imshow(channel_map, cmap='viridis')
                axes[i].set_title(f'Channel {i}')
                axes[i].axis('off')
            
            # Hide unused subplots
            for i in range(num_channels, len(axes)):
                axes[i].axis('off')
            
            plt.tight_layout()
            
            if save_path:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                plt.savefig(f"{save_path}_{layer_name.replace('.', '_')}.png")
                print(f"Feature map visualization saved to {save_path}_{layer_name.replace('.', '_')}.png")
            
            plt.close()
    
    return feature_maps