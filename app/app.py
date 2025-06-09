import streamlit as st
import os
import sys
import pandas as pd
from PIL import Image
import plotly.express as px

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from models.dinov2_encoder import DinoV2Encoder
from models.clip_encoder import CLIPEncoder
from search.search_engine import ImageSearchEngine
from utils.config import Config

@st.cache_resource
def load_search_engine(model_type: str, index_path: str = None):
    """
    Load and cache search engine
    """
    if model_type == 'dinov2':
        encoder = DinoV2Encoder
    elif model_type == 'clip':
        encoder = CLIPEncoder
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    engine = ImageSearchEngine(encoder)

    if index_path and os.path.exists(f"{index_path}.index"):
        engine.load_index(index_path)

def main():
    st.set_page_config(
        page_title="Image Similarity Search",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Image Similarity Search Engine")
    st.markdown("Upload an image to find visually similar images in the database")

    # Sidebar configuration
    with st.sidebar:
        st.header("Configuration")

        model_type = st.selectbox(
            "Select Model",
            ["dinov2", "clip"],
            help="Choose the embedding model"
        )

        index_path = st.text_input(
            "Index Path",
            value="data/embeddings/index",
            help="Path to the pre-built search index"
        )

        k = st.slider("Number of results", 1, 50, 10)

        # Load Search Engine
        try:
            engine = load_search_engine(model_type, index_path)
            stats = engine.get_stats()

            st.success(f"Model loaded: {stats['model_name']}")
            st.info(f"Database: {stats['total_images']} images")
            
        except Exception as e:
            st.error(f"Failed to load model: {e}")
            return
    
    # Main interface
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Query Image")
        
        uploaded_file = st.file_uploader(
            "Upload an image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload an image to search for similar images"
        )
        
        if uploaded_file:
            query_image = Image.open(uploaded_file)
            st.image(query_image, caption="Query Image", use_column_width=True)
            
            if st.button("Search Similar Images", type="primary"):
                with st.spinner("Searching..."):
                    try:
                        results = engine.search_by_image(query_image, k=k)
                        st.session_state['search_results'] = results
                    except Exception as e:
                        st.error(f"Search failed: {e}")
    
    with col2:
        st.header("Search Results")
        
        if 'search_results' in st.session_state:
            results = st.session_state['search_results']
            
            # Display search statistics
            if results:
                col2a, col2b, col2c = st.columns(3)
                col2a.metric("Results Found", len(results))
                col2b.metric("Encode Time", f"{results[0].get('encode_time', 0):.3f}s")
                col2c.metric("Search Time", f"{results[0].get('search_time', 0):.3f}s")
            
            # Display results
            for i, result in enumerate(results):
                with st.container():
                    col_img, col_info = st.columns([1, 2])
                    
                    with col_img:
                        try:
                            img_path = result['metadata']['path']
                            if os.path.exists(img_path):
                                img = Image.open(img_path)
                                st.image(img, use_column_width=True)
                            else:
                                st.error("Image not found")
                        except Exception as e:
                            st.error(f"Error loading image: {e}")
                    
                    with col_info:
                        st.write(f"**Rank:** {result['rank']}")
                        st.write(f"**Similarity:** {result['similarity']:.4f}")
                        st.write(f"**Distance:** {result['distance']:.4f}")
                        st.write(f"**Filename:** {result['metadata']['filename']}")
                
                st.divider()
        
        else:
            st.info("Upload an image and click 'Search' to see results")

if __name__ == "__main__":
    main()