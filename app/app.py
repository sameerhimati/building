import streamlit as st
import torch
import os
import sys
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import time
from huggingface_hub import hf_hub_download

# Add the project root to the path so we can import from src
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import from our existing modules
from src.dataset import get_data_transforms
from src.models import create_efficientnet_model  # Assuming this function exists in your models.py

# Set page configuration
st.set_page_config(
    page_title="What's That Building?",
    page_icon="🏛️",
    layout="wide"
)

def load_model_from_hf(repo_id, filename):
    """Load model from Hugging Face Hub."""
    try:
        # Download model file
        model_path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
        )
        return model_path
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return None

def find_model_paths():
    """Find all available model paths (local or remote)."""
    models = []

    models.append((
        "🚀 Fine Tuned EfficientNetV2 Model (Hugging Face)", 
        "https://huggingface.co/sameerhimati/architectural-style-classifier-EfficientNetFineTuned/resolve/main/best_model_fine_tuned.pth"
    ))
    
    # Look in outputs directory (local development)
    output_dir = "outputs"
    if os.path.exists(output_dir):
        for run_dir in sorted(os.listdir(output_dir), reverse=True):
            run_path = os.path.join(output_dir, run_dir)
            if os.path.isdir(run_path):
                # Check for fine-tuned model first
                fine_tuned_path = os.path.join(run_path, "best_model_fine_tuned.pth")
                base_model_path = os.path.join(run_path, "best_model.pth")
                
                if os.path.exists(fine_tuned_path):
                    models.append(("✨ " + run_dir + " (fine-tuned)", fine_tuned_path))
                
                if os.path.exists(base_model_path):
                    models.append((run_dir, base_model_path))
    
    # Add remote models (for deployment)
    # Example: models.append(("Production Model v1", "https://storage-url.com/model.pth"))
    
    # If no models found, add placeholder
    if not models:
        models.append(("Demo Mode - No models available", "demo"))
    
    return models

# Update load_model function to handle remote or missing models
@st.cache_resource
def load_model(model_path):
    """Load the trained model from local path or remote URL."""
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Demo mode - return pretrained model
    if model_path == "demo":
        st.warning("⚠️ Running in demo mode with a pretrained model. Accuracy may be limited.")
        num_classes = 45  # Your number of classes
        model = create_efficientnet_model(num_classes, pretrained=True, freeze_backbone=False)
        class_names = [f"Architectural Style {i+1}" for i in range(num_classes)]
        return model, device, class_names
    
    # Hugging Face
    if model_path.startswith("huggingface:"):
        _, repo_id, filename = model_path.split(":")
        local_path = load_model_from_hf(repo_id, filename)
        if local_path:
            model_path = local_path
        else:
            st.error("Failed to load model from Hugging Face")

    # Remote URL
    elif model_path.startswith(("http://", "https://")):
        st.info("Downloading model from remote location...")
        # Download to temp file
        import tempfile
        import requests
        
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            response = requests.get(model_path, stream=True)
            response.raise_for_status()
            for chunk in response.iter_content(chunk_size=8192):
                tmp.write(chunk)
            
            checkpoint_path = tmp.name
    else:
        checkpoint_path = model_path
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model state dict and class names
    if 'model_state_dict' in checkpoint:
        model_state_dict = checkpoint['model_state_dict']
        class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(45)])
    else:
        model_state_dict = checkpoint
        class_names = [f"Class_{i}" for i in range(45)]
    
    # Create model with same architecture
    num_classes = len(class_names)
    model = create_efficientnet_model(num_classes, pretrained=False, freeze_backbone=False)
    
    # Load weights
    model.load_state_dict(model_state_dict)
    model.to(device)
    model.eval()  # Set to evaluation mode
    
    return model, device, class_names

def visualize_top_classes(probabilities, class_indices, class_names, image, top_k=3):
    """Visualize top-k class probabilities."""
    plt.figure(figsize=(12, 5))
    
    # Plot the image on the left
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.axis('off')
    plt.title("Building Image")
    
    # Plot the top-k probabilities on the right
    plt.subplot(1, 2, 2)
    
    # Get class names and probabilities
    top_classes = [class_names[idx] for idx in class_indices]
    top_probs = probabilities * 100  # Convert to percentage
    
    # Create horizontal bar chart
    bars = plt.barh(range(top_k), top_probs, color='skyblue')
    plt.yticks(range(top_k), top_classes)
    plt.xlabel('Probability (%)')
    plt.title('Top Architectural Styles')
    
    # Add probability labels
    for i, bar in enumerate(bars):
        plt.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, 
                 f'{top_probs[i]:.1f}%', va='center')
    
    plt.tight_layout()
    return plt

def main():
    # Page header with logo and title
    col1, col2 = st.columns([1, 4])
    with col1:
        st.image("app/utils/skyscraper.png", width=100)
    with col2:
        st.title("What's That Building?")
        st.subheader("Architectural Style Classifier")
    
    st.markdown("""
    Upload an image of a building to identify its architectural style. 
    Our classifier can recognize 45 different architectural styles from around the world.
    """)
    
    # Sidebar for model selection
    st.sidebar.title("Model Settings")
    
    # Get available models
    available_models = find_model_paths()
    
    if not available_models:
        st.error("No trained models found in the outputs directory!")
        return
        
    # Display dropdown with model options
    model_options = {name: path for name, path in available_models}
    selected_model_name = st.sidebar.selectbox(
        "Select Model",
        list(model_options.keys()),
        index=0  # Default to first (newest) model
    )
    
    model_path = model_options[selected_model_name]
    
    # Display model details
    st.sidebar.caption(f"Path: {model_path}")
    
    # Number of predictions to show
    top_k = st.sidebar.slider("Number of predictions to show", 1, 5, 3)
    
    # Confidence threshold
    confidence_threshold = st.sidebar.slider(
        "Minimum confidence threshold (%)", 
        0, 100, 10
    ) / 100.0
    
    # Load model
    with st.spinner("Loading model..."):
        try:
            model, device, class_names = load_model(model_path)
            st.sidebar.success(f"Model loaded successfully with {len(class_names)} architectural styles")
        except Exception as e:
            st.sidebar.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Display tabs for different input methods
    tab1, tab2 = st.tabs(["Upload Image", "Take Photo"])
    
    with tab1:
        # File uploader
        uploaded_file = st.file_uploader(
            "Upload an image of a building", 
            type=["jpg", "jpeg", "png"]
        )
        image_file = uploaded_file
        
    with tab2:
        # Camera input
        camera_input = st.camera_input("Take a photo of a building")
        if camera_input:
            image_file = camera_input
    
    # Process image if available
    if image_file is not None:
        # Display the image
        image = Image.open(image_file)
        st.image(image, caption="Building Image", use_container_width=True)
        
        # Preprocess the image
        transforms = get_data_transforms()['val']  # Use validation transforms
        input_tensor = transforms(image).unsqueeze(0)
        
        # Make prediction
        with st.spinner("Analyzing architectural style..."):
            # Simulate a slight delay to show the spinner
            time.sleep(0.5)
            
            with torch.no_grad():
                input_tensor = input_tensor.to(device)
                outputs = model(input_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
                
                # Get top-k predictions
                top_probs, top_idx = torch.topk(probabilities, k=top_k)
        
        # Convert to numpy for visualization
        top_probs_np = top_probs.cpu().numpy()
        top_idx_np = top_idx.cpu().numpy()
        
        # Create visualization
        fig = visualize_top_classes(
            top_probs_np, top_idx_np, class_names, image, top_k
        )
        st.pyplot(fig)
        
        # Show result details in expander
        with st.expander("See detailed analysis"):
            st.write("### Top Architectural Styles")
            
            # Create a table of results
            for i in range(top_k):
                class_idx = top_idx_np[i]
                class_name = class_names[class_idx]
                probability = top_probs_np[i] * 100
                probability = float(probability)
                
                st.write(f"**{i+1}. {class_name}**")
                st.progress(probability / 100)
                st.write(f"Confidence: {probability:.2f}%")
                
                if probability < confidence_threshold * 100:
                    st.warning(f"Low confidence prediction (below {confidence_threshold*100}%)")
        
        # Low confidence warning
        if top_probs_np[0] < confidence_threshold:
            st.warning("""
            ⚠️ **Low confidence in the top prediction.** 
            
            The architectural style might be ambiguous or not well-represented in our training data.
            Consider trying a different angle or a clearer image of the building.
            """)

if __name__ == "__main__":
    main()