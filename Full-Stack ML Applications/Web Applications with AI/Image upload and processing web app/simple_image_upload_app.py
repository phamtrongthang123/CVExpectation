#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "streamlit",
#     "torch",
#     "torchvision",
#     "pillow",
#     "numpy",
# ]
# ///

"""
Simple image upload and processing web app using Streamlit.
Demonstrates: file upload, image processing, and real-time predictions.

Run with: uv run simple_image_upload_app.py
Then open: http://localhost:8501
"""

import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Configure page
st.set_page_config(page_title="Image Classifier", layout="wide")

# Load model (cached)
@st.cache_resource
def load_model():
    """Load pretrained ResNet model"""
    model = models.resnet18(pretrained=True)
    model.eval()
    return model

@st.cache_data
def load_labels():
    """Load ImageNet labels"""
    # Simplified labels
    return [f"class_{i}" for i in range(1000)]

# Image preprocessing
def preprocess_image(image):
    """Preprocess image for model"""
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(image, model):
    """Make prediction on image"""
    input_tensor = preprocess_image(image)

    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top 5 predictions
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    results = []
    for i in range(5):
        results.append({
            'class_id': top5_catid[i].item(),
            'probability': top5_prob[i].item()
        })

    return results

# Header
st.title("🖼️ Image Classification Web App")
st.markdown("Upload an image and get predictions from a pretrained ResNet model")
st.markdown("---")

# Load model
model = load_model()
labels = load_labels()

# Sidebar
with st.sidebar:
    st.header("Settings")
    show_preprocessed = st.checkbox("Show preprocessed image", value=False)
    confidence_threshold = st.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.05)

# Main content
col1, col2 = st.columns(2)

with col1:
    st.header("Upload Image")

    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file).convert('RGB')

        # Display original image
        st.image(image, caption='Uploaded Image', use_container_width=True)

        # Show image info
        st.info(f"Image size: {image.size[0]}x{image.size[1]} pixels")

        # Show preprocessed image if requested
        if show_preprocessed:
            preprocessed = preprocess_image(image)
            # Convert back to displayable format
            display_tensor = preprocessed.squeeze(0)
            # Denormalize
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            display_tensor = display_tensor * std + mean
            display_tensor = torch.clamp(display_tensor, 0, 1)

            st.image(display_tensor.permute(1, 2, 0).numpy(), caption='Preprocessed Image', use_container_width=True)

with col2:
    st.header("Predictions")

    if uploaded_file is not None:
        # Make prediction
        with st.spinner('Classifying...'):
            results = predict(image, model)

        # Display results
        st.success("Classification complete!")

        for i, result in enumerate(results):
            class_name = labels[result['class_id']]
            prob = result['probability']

            # Color code based on confidence
            if prob > confidence_threshold:
                st.markdown(f"**{i+1}. {class_name}**")
                st.progress(prob)
                st.caption(f"Confidence: {prob*100:.2f}%")
            else:
                st.markdown(f"{i+1}. {class_name}")
                st.progress(prob)
                st.caption(f"Confidence: {prob*100:.2f}% (below threshold)")

            st.markdown("")

    else:
        st.info("👆 Upload an image to see predictions")

st.markdown("---")

# Example images section
st.header("📸 Example Images")
st.markdown("Try these example images or upload your own!")

example_col1, example_col2, example_col3 = st.columns(3)

with example_col1:
    st.markdown("**Cat**")
    st.caption("Upload a cat image to classify")

with example_col2:
    st.markdown("**Dog**")
    st.caption("Upload a dog image to classify")

with example_col3:
    st.markdown("**Car**")
    st.caption("Upload a car image to classify")

# Footer
st.markdown("---")
st.caption("Powered by PyTorch and Streamlit")
