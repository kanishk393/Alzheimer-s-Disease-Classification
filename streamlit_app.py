"""
Streamlit web application for Alzheimer's disease classification.

This application provides a user-friendly interface for uploading MRI images
and getting predictions from the trained model.
"""

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import os
import sys
import yaml
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from alzheimer_classifier.models.efficientnet_model import load_model


# Page configuration
st.set_page_config(
    page_title="Alzheimer's Disease Classification",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .subheader {
        font-size: 1.5rem;
        color: #ff7f0e;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f0f8ff;
        border: 2px solid #1f77b4;
        border-radius: 10px;
        padding: 20px;
        margin: 20px 0;
    }
    .confidence-high {
        color: #2e8b57;
        font-weight: bold;
    }
    .confidence-medium {
        color: #ff8c00;
        font-weight: bold;
    }
    .confidence-low {
        color: #dc143c;
        font-weight: bold;
    }
    .info-box {
        background-color: #f8f9fa;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Class mappings
CLASS_NAMES = {
    0: "Mild Demented",
    1: "Moderate Demented", 
    2: "Non Demented",
    3: "Very Mild Demented"
}

CLASS_DESCRIPTIONS = {
    "Non Demented": {
        "description": "No signs of dementia detected. Brain function appears normal.",
        "color": "#2e8b57",
        "recommendations": [
            "Continue regular health check-ups",
            "Maintain healthy lifestyle",
            "Engage in cognitive activities"
        ]
    },
    "Very Mild Demented": {
        "description": "Very early stage of dementia. Slight memory issues may be present.",
        "color": "#ffa500",
        "recommendations": [
            "Consult with a neurologist",
            "Consider cognitive exercises",
            "Monitor symptoms closely"
        ]
    },
    "Mild Demented": {
        "description": "Mild stage of dementia. Noticeable memory and cognitive issues.",
        "color": "#ff6347",
        "recommendations": [
            "Seek professional medical evaluation",
            "Discuss treatment options",
            "Plan for future care needs"
        ]
    },
    "Moderate Demented": {
        "description": "Moderate stage of dementia. Significant cognitive impairment.",
        "color": "#dc143c",
        "recommendations": [
            "Immediate medical consultation required",
            "Consider specialized care",
            "Discuss safety measures"
        ]
    }
}


@st.cache_resource
def load_model_cached(model_path):
    """Load model with caching to avoid reloading."""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model(model_path, device=device)
        return model, device
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


def preprocess_image(image):
    """Preprocess uploaded image for model inference."""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    tensor = transform(image).unsqueeze(0)
    return tensor


def predict_image(model, image_tensor, device):
    """Make prediction on preprocessed image."""
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        
        # Get predictions
        logits = model(image_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
        confidence = probabilities[0][predicted_class].item()
        
        # Get all class probabilities
        all_probs = probabilities[0].cpu().numpy()
        
    return predicted_class, confidence, all_probs


def create_probability_chart(probabilities):
    """Create a bar chart showing prediction probabilities."""
    df = pd.DataFrame({
        'Class': [CLASS_NAMES[i] for i in range(len(probabilities))],
        'Probability': probabilities
    })
    
    fig = px.bar(
        df, 
        x='Class', 
        y='Probability',
        title='Prediction Probabilities',
        color='Probability',
        color_continuous_scale='RdYlBu_r'
    )
    
    fig.update_layout(
        xaxis_title="Dementia Stage",
        yaxis_title="Probability",
        showlegend=False,
        height=400
    )
    
    return fig


def display_class_info(predicted_class, confidence):
    """Display information about the predicted class."""
    class_name = CLASS_NAMES[predicted_class]
    class_info = CLASS_DESCRIPTIONS[class_name]
    
    # Confidence color based on value
    if confidence > 0.8:
        conf_class = "confidence-high"
        conf_text = "High"
    elif confidence > 0.6:
        conf_class = "confidence-medium"
        conf_text = "Medium"
    else:
        conf_class = "confidence-low"
        conf_text = "Low"
    
    st.markdown(f"""
    <div class="prediction-box">
        <h3>🎯 Prediction Results</h3>
        <h2 style="color: {class_info['color']};">{class_name}</h2>
        <p><strong>Confidence:</strong> <span class="{conf_class}">{confidence:.2%} ({conf_text})</span></p>
        <p><strong>Description:</strong> {class_info['description']}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Recommendations
    st.markdown(f"""
    <div class="info-box">
        <h4>💡 Recommendations</h4>
        <ul>
            {"".join([f"<li>{rec}</li>" for rec in class_info['recommendations']])}
        </ul>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">🧠 Alzheimer\'s Disease Classification</h1>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div class="info-box">
        <p>This application uses a deep learning model to classify MRI brain scans for Alzheimer's disease detection. 
        Upload an MRI image to get predictions about the dementia stage.</p>
        <p><strong>⚠️ Disclaimer:</strong> This tool is for educational purposes only and should not be used as a 
        substitute for professional medical diagnosis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.markdown("## 🔧 Settings")
    
    # Model selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="models/final/best_model.pth",
        help="Path to the trained model file"
    )
    
    # Load model
    if os.path.exists(model_path):
        model, device = load_model_cached(model_path)
        if model is not None:
            st.sidebar.success("✅ Model loaded successfully!")
            st.sidebar.info(f"Device: {device}")
        else:
            st.sidebar.error("❌ Failed to load model")
            return
    else:
        st.sidebar.error("❌ Model file not found")
        st.sidebar.info("Please ensure the model file exists at the specified path")
        return
    
    # File uploader
    st.markdown('<h2 class="subheader">📤 Upload MRI Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an MRI image...",
        type=['jpg', 'jpeg', 'png', 'bmp', 'tiff'],
        help="Upload a brain MRI scan image for classification"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### 🖼️ Uploaded Image")
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded MRI Scan", use_column_width=True)
            
            # Image info
            st.markdown(f"""
            <div class="info-box">
                <p><strong>Image Size:</strong> {image.size[0]} x {image.size[1]} pixels</p>
                <p><strong>Format:</strong> {image.format}</p>
                <p><strong>Mode:</strong> {image.mode}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Prediction button
            if st.button("🚀 Analyze Image", type="primary"):
                with st.spinner("Analyzing image..."):
                    # Preprocess image
                    image_tensor = preprocess_image(image)
                    
                    # Make prediction
                    predicted_class, confidence, all_probs = predict_image(
                        model, image_tensor, device
                    )
                    
                    # Display results
                    display_class_info(predicted_class, confidence)
                    
                    # Probability chart
                    fig = create_probability_chart(all_probs)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Detailed probabilities
                    st.markdown("### 📊 Detailed Probabilities")
                    prob_df = pd.DataFrame({
                        'Class': [CLASS_NAMES[i] for i in range(len(all_probs))],
                        'Probability': [f"{prob:.2%}" for prob in all_probs]
                    })
                    st.dataframe(prob_df, use_container_width=True)
    
    # Information section
    st.markdown("---")
    st.markdown('<h2 class="subheader">📋 About the Model</h2>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="info-box">
            <h4>🏗️ Architecture</h4>
            <ul>
                <li><strong>Model:</strong> EfficientNet-B0</li>
                <li><strong>Input Size:</strong> 224 x 224 pixels</li>
                <li><strong>Classes:</strong> 4 dementia stages</li>
                <li><strong>Parameters:</strong> ~5.3M</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>📈 Performance</h4>
            <ul>
                <li><strong>Test Accuracy:</strong> 99.73%</li>
                <li><strong>Validation Accuracy:</strong> 99.76%</li>
                <li><strong>Training Time:</strong> 2.5 hours</li>
                <li><strong>Inference Time:</strong> ~45ms</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="info-box">
            <h4>🔬 Dataset</h4>
            <ul>
                <li><strong>Total Images:</strong> 6,400</li>
                <li><strong>Source:</strong> Kaggle</li>
                <li><strong>Augmentation:</strong> Yes</li>
                <li><strong>Preprocessing:</strong> Standardized</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Class information
    st.markdown('<h2 class="subheader">🧭 Dementia Stages</h2>', unsafe_allow_html=True)
    
    for class_name, info in CLASS_DESCRIPTIONS.items():
        st.markdown(f"""
        <div class="info-box">
            <h4 style="color: {info['color']};">{class_name}</h4>
            <p>{info['description']}</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <p>🧠 Alzheimer's Disease Classification System | Built with ❤️ using Streamlit and PyTorch</p>
        <p>For research and educational purposes only</p>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()