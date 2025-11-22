#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "gradio",
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple interactive demo using Gradio for research/model demonstration.
Demonstrates: interactive UI, model inference, and visualization.

Run with: uv run simple_interactive_demo.py
"""

import gradio as gr
import torch
import torch.nn as nn
import numpy as np

# Simple model for demo
class DemoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)

# Load model
model = DemoModel()
model.eval()

# Prediction function
def predict(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    """Make prediction from 10 input features"""
    # Combine inputs
    features = torch.tensor([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dtype=torch.float32)

    # Normalize
    features = (features - features.mean()) / (features.std() + 1e-8)

    # Predict
    with torch.no_grad():
        output = model(features.unsqueeze(0))
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = output.argmax(dim=1).item()

    # Format output
    results = {
        f"Class {i}": float(probabilities[i]) for i in range(3)
    }

    confidence = float(probabilities[prediction])

    return f"Predicted Class: {prediction}", results, f"Confidence: {confidence:.2%}"

# Visualization function
def visualize_features(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10):
    """Visualize feature importance (simple demo)"""
    features = np.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10])

    # Calculate simple importance (absolute value)
    importance = np.abs(features)
    normalized_importance = importance / (importance.sum() + 1e-8)

    return {
        f"Feature {i+1}": float(normalized_importance[i]) for i in range(10)
    }

# Create Gradio interface
with gr.Blocks(title="Research Demo", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🧪 Interactive Research Demo")
    gr.Markdown("Demonstrate your ML model with an interactive interface")

    with gr.Tab("Prediction"):
        gr.Markdown("### Input Features")
        gr.Markdown("Adjust the sliders to change input values and see predictions")

        with gr.Row():
            with gr.Column():
                inputs = [
                    gr.Slider(-5, 5, value=0, label=f"Feature {i+1}", step=0.1)
                    for i in range(10)
                ]

            with gr.Column():
                prediction_output = gr.Textbox(label="Prediction", interactive=False)
                confidence_output = gr.Textbox(label="Confidence", interactive=False)
                probabilities_output = gr.Label(label="Class Probabilities", num_top_classes=3)

        predict_btn = gr.Button("Predict", variant="primary")

        predict_btn.click(
            fn=predict,
            inputs=inputs,
            outputs=[prediction_output, probabilities_output, confidence_output]
        )

    with gr.Tab("Feature Analysis"):
        gr.Markdown("### Feature Importance")
        gr.Markdown("See which features are most important for this prediction")

        with gr.Row():
            with gr.Column():
                analysis_inputs = [
                    gr.Slider(-5, 5, value=0, label=f"Feature {i+1}", step=0.1)
                    for i in range(10)
                ]

            with gr.Column():
                importance_output = gr.Label(label="Feature Importance", num_top_classes=10)

        analyze_btn = gr.Button("Analyze", variant="primary")

        analyze_btn.click(
            fn=visualize_features,
            inputs=analysis_inputs,
            outputs=importance_output
        )

    with gr.Tab("About"):
        gr.Markdown("""
        ### About This Demo

        This is a simple interactive demo showcasing:
        - Real-time model inference
        - Interactive parameter adjustment
        - Feature importance visualization

        **Model Architecture:**
        - Input: 10 features
        - Hidden layers: 2 × 64 neurons
        - Output: 3 classes

        **How to Use:**
        1. Go to the "Prediction" tab
        2. Adjust the feature sliders
        3. Click "Predict" to see results
        4. Check "Feature Analysis" to understand feature importance

        **Technologies:**
        - Gradio for the interface
        - PyTorch for the model
        - NumPy for calculations
        """)

    gr.Markdown("---")
    gr.Markdown("*Powered by Gradio and PyTorch*")

# Launch the demo
if __name__ == "__main__":
    demo.launch(share=False, server_name="0.0.0.0", server_port=7860)
