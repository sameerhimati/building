import torch
import os
import sys
import shutil
import argparse

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def package_model(model_path, output_name):
    """Package a trained model for inclusion with the app."""
    # Create models directory if it doesn't exist
    os.makedirs("app/models", exist_ok=True)
    
    # Load the checkpoint
    checkpoint = torch.load(model_path, map_location="cpu")
    
    # Save it to the models directory
    output_path = f"app/models/{output_name}.pth"
    torch.save(checkpoint, output_path)
    
    print(f"Model packaged successfully: {output_path}")
    print(f"File size: {os.path.getsize(output_path) / (1024 * 1024):.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Package a model for the Streamlit app")
    parser.add_argument("--model_path", required=True, help="Path to the model checkpoint")
    parser.add_argument("--output_name", required=True, help="Name for the packaged model")
    args = parser.parse_args()
    
    package_model(args.model_path, args.output_name)