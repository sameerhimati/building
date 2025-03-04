import os
import json
import datetime
import uuid
import streamlit as st
from PIL import Image
import io

class FeedbackCollector:
    """
    Collect user feedback and images for continual learning.
    """
    def __init__(self, base_dir="feedback"):
        # Create feedback directory
        self.base_dir = base_dir
        self.images_dir = os.path.join(base_dir, "images")
        self.feedback_file = os.path.join(base_dir, "feedback.json")
        
        # Ensure directories exist
        os.makedirs(self.base_dir, exist_ok=True)
        os.makedirs(self.images_dir, exist_ok=True)
        
        # Initialize feedback data
        self.feedback_data = self._load_feedback()
    
    def _load_feedback(self):
        """Load existing feedback data or create new structure."""
        if os.path.exists(self.feedback_file):
            try:
                with open(self.feedback_file, 'r') as f:
                    return json.load(f)
            except:
                return {"feedback": [], "stats": {"total": 0, "correct": 0, "incorrect": 0}}
        else:
            return {"feedback": [], "stats": {"total": 0, "correct": 0, "incorrect": 0}}
    
    def _save_feedback(self):
        """Save feedback data to JSON file."""
        with open(self.feedback_file, 'w') as f:
            json.dump(self.feedback_data, f, indent=2)
    
    def save_image(self, image):
        """
        Save user-uploaded image for future training.
        
        Args:
            image: PIL Image object
            
        Returns:
            str: Unique image filename
        """
        # Generate unique filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        filename = f"{timestamp}_{unique_id}.jpg"
        
        # Save image
        image_path = os.path.join(self.images_dir, filename)
        image.save(image_path)
        
        return filename
    
    def collect_feedback(self, image, predicted_style, correct_style=None, is_correct=None):
        """
        Collect user feedback for a prediction.
        
        Args:
            image: PIL Image object
            predicted_style (str): Model's prediction
            correct_style (str, optional): User-provided correct style if model was wrong
            is_correct (bool, optional): Whether prediction was correct
            
        Returns:
            dict: Feedback entry
        """
        # Save the image
        image_filename = self.save_image(image)
        
        # Record feedback
        feedback_entry = {
            "id": str(uuid.uuid4()),
            "timestamp": datetime.datetime.now().isoformat(),
            "image_filename": image_filename,
            "predicted_style": predicted_style,
            "is_correct": is_correct,
            "correct_style": correct_style if not is_correct else predicted_style
        }
        
        # Update statistics
        self.feedback_data["stats"]["total"] += 1
        if is_correct:
            self.feedback_data["stats"]["correct"] += 1
        else:
            self.feedback_data["stats"]["incorrect"] += 1
        
        # Add to feedback list
        self.feedback_data["feedback"].append(feedback_entry)
        
        # Save updated data
        self._save_feedback()
        
        return feedback_entry
    
    def get_stats(self):
        """Get feedback statistics."""
        stats = self.feedback_data["stats"]
        
        # Calculate accuracy
        if stats["total"] > 0:
            stats["accuracy"] = stats["correct"] / stats["total"] * 100
        else:
            stats["accuracy"] = 0
            
        return stats
    
    def display_stats(self):
        """Display feedback statistics in Streamlit."""
        stats = self.get_stats()
        
        st.subheader("User Feedback Statistics")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Predictions", stats["total"])
        with col2:
            st.metric("Correct Predictions", stats["correct"])
        with col3:
            st.metric("Accuracy", f"{stats.get('accuracy', 0):.1f}%")