import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def evaluate_model(model, test_loader, criterion, device, class_names):
    """
    Evaluate model performance on the test set
    
    Args:
        model: PyTorch model
        test_loader: DataLoader for test dataset
        criterion: Loss function
        device: Device to evaluate on
        class_names: List of class names
        
    Returns:
        tuple: (test_loss, test_accuracy, predictions, true_labels)
    """
    model.eval()
    test_loss = 0.0
    test_corrects = 0
    all_preds = []
    all_labels = []
    
    # No gradient needed for evaluation
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Update statistics
            test_loss += loss.item() * inputs.size(0)
            test_corrects += torch.sum(preds == labels.data).item()
            
            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    test_loss = test_loss / len(test_loader.dataset)
    test_acc = test_corrects / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    return test_loss, test_acc, np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(predictions, true_labels, class_names, save_path=None):
    """
    Plot confusion matrix
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        class_names: List of class names
        save_path: Path to save the plot to (optional)
    """
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Normalize by row (true labels)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Get the number of classes
    n_classes = len(class_names)
    
    # If too many classes, use a subset for clarity
    if n_classes > 20:
        # Find top 15 classes by frequency
        class_counts = np.sum(cm, axis=1)
        top_indices = np.argsort(class_counts)[-15:]
        
        # Extract submatrix and class names
        cm_norm = cm_norm[top_indices][:, top_indices]
        displayed_classes = [class_names[i] for i in top_indices]
        print(f"Showing top 15 classes out of {n_classes} total classes")
    else:
        displayed_classes = class_names
    
    # Plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=displayed_classes, yticklabels=displayed_classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Normalized Confusion Matrix')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print(f'Confusion matrix saved to {save_path}')
    
    plt.show()

    
def print_classification_report(predictions, true_labels, class_names):
    """
    Print classification report with precision, recall, and f1-score
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        class_names: List of class names
    """
    report = classification_report(true_labels, predictions, 
                                  target_names=class_names, digits=3)
    print("Classification Report:")
    print(report)


def identify_difficult_classes(predictions, true_labels, class_names, top_n=5):
    """
    Identify the most challenging classes for the model
    
    Args:
        predictions: Model predictions
        true_labels: True labels
        class_names: List of class names
        top_n: Number of most difficult classes to report
        
    Returns:
        list: Class indices sorted by error rate (most difficult first)
    """
    # Create confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Calculate accuracy per class
    class_acc = np.diag(cm) / np.sum(cm, axis=1)
    
    # Get indices of classes sorted by accuracy (ascending)
    difficult_class_indices = np.argsort(class_acc)
    
    print(f"Top {top_n} most difficult classes:")
    for i in range(min(top_n, len(class_names))):
        idx = difficult_class_indices[i]
        print(f"{i+1}. {class_names[idx]}: {class_acc[idx]:.3f} accuracy")
    
    return difficult_class_indices