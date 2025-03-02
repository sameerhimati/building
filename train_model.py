import torch
import argparse
import os
import time
from datetime import datetime

# Import our modules
from src.dataset import get_data_transforms, create_datasets, create_dataloaders
from src.models import create_efficientnet_model
from src.training import setup_training, train_model, plot_training_history
from src.evaluation import evaluate_model, plot_confusion_matrix, print_classification_report, identify_difficult_classes

def parse_args():
    parser = argparse.ArgumentParser(description="Train architectural style classifier")
    parser.add_argument("--data_dir", type=str, default="data/processed", 
                        help="Path to dataset directory")
    
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size for training")
    
    parser.add_argument("--num_epochs", type=int, default=10, 
                        help="Number of epochs to train for")
    
    parser.add_argument("--learning_rate", type=float, default=0.001, 
                        help="Initial learning rate")
    
    parser.add_argument("--custom_stats", action="store_true", 
                        help="Calculate custom normalization statistics")
    
    parser.add_argument("--output_dir", type=str, default="outputs", 
                        help="Directory to save outputs")
    
    return parser.parse_args()

def main():
    # Parse arguments
    args = parse_args()
    
    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Log training parameters
    with open(os.path.join(output_dir, "training_params.txt"), "w") as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")
    
    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Using device: {device}")
    
    # Get data transforms
    transforms = get_data_transforms(args.data_dir, args.custom_stats)
    
    # Create datasets
    train_dataset, val_dataset, test_dataset, class_names = create_datasets(
        args.data_dir, transforms
    )
    
    # Create dataloaders
    dataloaders = {}
    dataloaders['train'], dataloaders['val'], test_loader = create_dataloaders(
        train_dataset, val_dataset, test_dataset, 
        batch_size=args.batch_size
    )
    
    # Create model
    num_classes = len(class_names)
    model = create_efficientnet_model(num_classes, pretrained=True, freeze_backbone=True)
    
    # Setup training components
    criterion, optimizer, scheduler = setup_training(
        model, device, learning_rate=args.learning_rate
    )
    
    # Train model
    print("Starting model training...")
    model, history = train_model(
        model, dataloaders, criterion, optimizer, scheduler, 
        device, num_epochs=args.num_epochs
    )
    
    # Save model
    model_path = os.path.join(output_dir, "best_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': class_names,
        'history': history
    }, model_path)
    print(f"Model saved to {model_path}")
    
    # Plot and save training history
    history_path = os.path.join(output_dir, "training_history.png")
    plot_training_history(history, save_path=history_path)
    
    # Evaluate model on test set
    print("\nEvaluating model on test set...")
    test_loss, test_acc, predictions, true_labels = evaluate_model(
        model, test_loader, criterion, device, class_names
    )
    
    # Create confusion matrix
    cm_path = os.path.join(output_dir, "confusion_matrix.png")
    plot_confusion_matrix(predictions, true_labels, class_names, save_path=cm_path)
    
    # Print classification report
    print_classification_report(predictions, true_labels, class_names)
    
    # Identify difficult classes
    identify_difficult_classes(predictions, true_labels, class_names, top_n=10)
    
    print(f"\nTraining complete! All outputs saved to {output_dir}")

if __name__ == "__main__":
    start_time = time.time()
    main()
    end_time = time.time()
    print(f"Total execution time: {(end_time - start_time)/60:.2f} minutes")