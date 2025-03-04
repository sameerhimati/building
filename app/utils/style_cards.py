import streamlit as st
import json
import os
from .wiki_integration import get_wiki_summary

# Cache the architectural style characteristics
@st.cache_data
def load_style_characteristics():
    """Load architectural style characteristics from JSON file or create default structure."""
    json_path = os.path.join("app", "utils", "architectural_styles.json")
    
    if os.path.exists(json_path):
        with open(json_path, 'r') as f:
            return json.load(f)
    else:
        # Return empty dictionary that will be populated over time
        return {}

def get_style_card(style_name):
    """
    Generate an information card for the architectural style.
    
    Args:
        style_name (str): Name of the architectural style
        
    Returns:
        dict: Style information including description, period, features, and examples
    """
    # Get style characteristics
    style_info = load_style_characteristics()
    
    # Get Wikipedia info
    wiki_info = get_wiki_summary(style_name, sentences=5)
    
    # Use stored characteristics if available, otherwise create defaults
    if style_name in style_info:
        info = style_info[style_name]
    else:
        info = {
            "period": "Unknown time period",
            "region": "Unknown region",
            "key_features": [
                "Information will be added as users provide feedback"
            ],
            "famous_examples": [
                "Information will be added as users provide feedback"
            ]
        }
    
    # Combine with Wikipedia info
    info["description"] = wiki_info["summary"]
    info["wiki_url"] = wiki_info["url"]
    info["image_url"] = wiki_info["image_url"]
    
    return info

def display_style_card(style_name, confidence):
    """
    Display an architectural style information card in Streamlit.
    
    Args:
        style_name (str): Name of the architectural style
        confidence (float): Model confidence percentage
    """
    info = get_style_card(style_name)
    
    # Create card with formatted style
    st.markdown(f"## {style_name}")
    st.markdown(f"*Confidence: {confidence:.1f}%*")
    
    # Layout with columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### Description")
        st.markdown(info["description"])
        
        st.markdown("### Key Features")
        features_html = "".join([f"- {feature}<br>" for feature in info["key_features"]])
        st.markdown(features_html, unsafe_allow_html=True)
        
        st.markdown(f"**Period**: {info['period']}")
        st.markdown(f"**Region**: {info['region']}")
        
        st.markdown(f"[Learn more on Wikipedia]({info['wiki_url']})")
    
    with col2:
        # Display image if available
        if info["image_url"]:
            st.image(info["image_url"], caption="Example", use_container_width=True)
        
        st.markdown("### Famous Examples")
        examples_html = "".join([f"- {example}<br>" for example in info["famous_examples"]])
        st.markdown(examples_html, unsafe_allow_html=True)