import requests
import streamlit as st
import wikipedia

@st.cache_data
def get_wiki_summary(style_name, sentences=3):
    """
    Fetch a summary from Wikipedia for the given architectural style.
    Uses st.cache_data for efficiency.
    
    Args:
        style_name (str): Name of the architectural style
        sentences (int): Number of sentences to return
        
    Returns:
        dict: Contains summary, url, and image_url if available
    """
    try:
        # Search for the page
        search_term = f"{style_name} architecture"
        page = wikipedia.page(search_term)
        
        # Get summary and URL
        summary = wikipedia.summary(search_term, sentences=sentences)
        url = page.url
        
        # Try to get an image if available
        image_url = None
        if page.images:
            for img in page.images:
                if any(ext in img.lower() for ext in ['.jpg', '.jpeg', '.png']):
                    image_url = img
                    break
        
        return {
            "summary": summary,
            "url": url,
            "image_url": image_url
        }
    except:
        # Fallback with more general search
        try:
            search_term = style_name
            page = wikipedia.page(search_term)
            summary = wikipedia.summary(search_term, sentences=sentences)
            url = page.url
            
            return {
                "summary": summary,
                "url": url,
                "image_url": None
            }
        except:
            return {
                "summary": f"No Wikipedia information found for {style_name}.",
                "url": f"https://en.wikipedia.org/wiki/Special:Search?search={style_name.replace(' ', '+')}+architecture",
                "image_url": None
            }