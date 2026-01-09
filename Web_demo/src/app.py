import streamlit as st
import torch
from PIL import Image
import numpy as np
import os
from model.loader import load_wsss_model
from model.inference import run_inference
from preprocessing.transforms import get_class_names, overlay_mask_on_image

# Page configuration
st.set_page_config(
    page_title="WSSS Inference Tool",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container background */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
    }
    
    /* Center titles */
    h1, h2, h3 {
        text-align: center;
        color: white;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.3);
    }
    
    /* Image containers */
    .stImage {
        border-radius: 15px;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
        overflow: hidden;
    }
    
    /* Info boxes */
    .info-box {
        background-color: white;
        border-radius: 10px;
        padding: 1.5rem;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Class badge */
    .class-badge {
        display: inline-block;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        margin: 0.2rem;
        font-size: 0.9rem;
        font-weight: bold;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: white;
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Welcome box */
    .welcome-box {
        text-align: center;
        padding: 4rem;
        background-color: rgba(255,255,255,0.95);
        border-radius: 15px;
        margin: 2rem;
        box-shadow: 0 8px 16px rgba(0,0,0,0.2);
    }
    
    .welcome-box h2 {
        color: #667eea;
        text-shadow: none;
    }
    
    .welcome-box p {
        color: #666;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'model' not in st.session_state:
    with st.spinner('🔄 Loading model...'):
        try:
            # Use relative path from src directory
            model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'model.pth')
            model_path = os.path.abspath(model_path)
            
            model, device = load_wsss_model(model_path)
            st.session_state.model = model
            st.session_state.device = str(device)
            st.session_state.model_loaded = True
        except Exception as e:
            st.error(f"❌ Error loading model: {str(e)}")
            st.session_state.model_loaded = False

# Title
st.markdown("<h1>🎨 WSSS Inference Tool</h1>", unsafe_allow_html=True)
st.markdown("<h3>Weakly Supervised Semantic Segmentation with DeepLabV3+</h3>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### 📁 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=["jpg", "jpeg", "png"],
        help="Upload an image for semantic segmentation"
    )
    
    st.markdown("---")
    
    # Model Info
    st.markdown("### 🤖 Model Information")
    if st.session_state.get('model_loaded', False):
        st.markdown(f"""
        <div class="info-box">
            <b>Architecture:</b> DeepLabV3+<br>
            <b>Encoder:</b> ResNet-101<br>
            <b>Classes:</b> 21 (Pascal VOC)<br>
            <b>Device:</b> {st.session_state.device.upper()}<br>
            <b>Input Size:</b> 384×384
        </div>
        """, unsafe_allow_html=True)
    else:
        st.error("Model not loaded")

# Main content area
if uploaded_file is not None:
    # Load image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Auto-run inference when image is uploaded
    if st.session_state.get('model_loaded', False):
        with st.spinner('🔍 Running segmentation...'):
            try:
                # Run inference
                colored_mask, segmentation_map = run_inference(
                    st.session_state.model,
                    image,
                    st.session_state.device
                )
                
                # Create overlay image
                overlay_image = overlay_mask_on_image(image, colored_mask, alpha=0.5)
                
                # Store results
                st.session_state.colored_mask = colored_mask
                st.session_state.segmentation_map = segmentation_map
                st.session_state.overlay_image = overlay_image
                
            except Exception as e:
                st.error(f"❌ Error during inference: {str(e)}")
    
    # Create three columns for display
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🖼️ Original Image")
        st.image(image, use_container_width=True, caption="Uploaded Image")
    
    with col2:
        if 'colored_mask' in st.session_state:
            st.markdown("### 🎯 Segmentation Mask")
            st.image(
                st.session_state.colored_mask,
                use_container_width=True,
                caption="Predicted Segmentation"
            )
    
    with col3:
        if 'overlay_image' in st.session_state:
            st.markdown("### 🔮 Overlay Result")
            st.image(
                st.session_state.overlay_image,
                use_container_width=True,
                caption="Mask Overlay (50% opacity)"
            )
    
    # Display detected classes with names
    if 'segmentation_map' in st.session_state:
        st.markdown("---")
        st.markdown("### 📊 Detection Results")
        
        unique_classes = np.unique(st.session_state.segmentation_map)
        class_names_dict = get_class_names()
        
        # Create badges for each detected class
        detected_classes_html = "<div style='background-color: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>"
        detected_classes_html += f"<p style='color: #666; margin-bottom: 1rem;'><b>Detected {len(unique_classes)} classes:</b></p>"
        
        for class_id in unique_classes:
            class_name = class_names_dict.get(int(class_id), f"Unknown ({class_id})")
            detected_classes_html += f'<span class="class-badge">{class_id}: {class_name}</span>'
        
        detected_classes_html += "</div>"
        st.markdown(detected_classes_html, unsafe_allow_html=True)
        
        # Show detailed statistics
        col_stat1, col_stat2 = st.columns(2)
        
        with col_stat1:
            total_pixels = st.session_state.segmentation_map.size
            st.markdown(f"""
            <div class="info-box">
                <b>Total Pixels:</b> {total_pixels:,}<br>
                <b>Image Size:</b> {st.session_state.segmentation_map.shape[1]} × {st.session_state.segmentation_map.shape[0]}
            </div>
            """, unsafe_allow_html=True)
        
        with col_stat2:
            # Calculate percentage for each class
            class_percentages = []
            for class_id in unique_classes:
                count = np.sum(st.session_state.segmentation_map == class_id)
                percentage = (count / total_pixels) * 100
                class_name = class_names_dict.get(int(class_id), f"Class {class_id}")
                class_percentages.append(f"{class_name}: {percentage:.2f}%")
            
            percentages_text = "<br>".join(class_percentages[:5])  # Show top 5
            if len(class_percentages) > 5:
                percentages_text += "<br>..."
            
            st.markdown(f"""
            <div class="info-box">
                <b>Class Coverage:</b><br>
                {percentages_text}
            </div>
            """, unsafe_allow_html=True)

else:
    # Welcome message when no image is uploaded
    st.markdown("""
    <div class="welcome-box">
        <h2>Welcome to WSSS Inference Tool! 👋</h2>
        <p style='font-size: 1.2rem;'>
            Upload an image using the sidebar to get started with semantic segmentation.
        </p>
        <p style='font-size: 1rem;'>
            This tool uses a DeepLabV3+ model trained on Pascal VOC dataset with 21 object classes.
        </p>
        <p style='font-size: 0.9rem; color: #888; margin-top: 1rem;'>
            ✨ Auto-inference enabled - results appear instantly after upload!
        </p>
    </div>
    """, unsafe_allow_html=True)