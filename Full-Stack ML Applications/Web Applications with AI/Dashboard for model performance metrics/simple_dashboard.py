#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "streamlit",
#     "numpy",
#     "pandas",
# ]
# ///

"""
Simple dashboard for model performance metrics using Streamlit.
Demonstrates: interactive visualization, metrics tracking, and web UI.

Run with: uv run simple_dashboard.py
Then open: http://localhost:8501
"""

import streamlit as st
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

# Configure page
st.set_page_config(page_title="ML Model Dashboard", layout="wide")

# Generate synthetic metrics data
@st.cache_data
def generate_metrics_data(days=30):
    """Generate synthetic training metrics"""
    dates = [datetime.now() - timedelta(days=x) for x in range(days, 0, -1)]

    data = {
        'date': dates,
        'train_loss': np.linspace(1.0, 0.2, days) + np.random.normal(0, 0.05, days),
        'val_loss': np.linspace(1.0, 0.25, days) + np.random.normal(0, 0.05, days),
        'train_acc': np.linspace(0.5, 0.95, days) + np.random.normal(0, 0.02, days),
        'val_acc': np.linspace(0.5, 0.92, days) + np.random.normal(0, 0.02, days),
    }

    return pd.DataFrame(data)

# Header
st.title("🤖 ML Model Performance Dashboard")
st.markdown("---")

# Metrics summary
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(label="Latest Train Accuracy", value="95.2%", delta="2.1%")

with col2:
    st.metric(label="Latest Val Accuracy", value="92.1%", delta="1.5%")

with col3:
    st.metric(label="Latest Val Loss", value="0.245", delta="-0.12")

with col4:
    st.metric(label="Total Predictions", value="1,245,891", delta="12,450")

st.markdown("---")

# Load data
df = generate_metrics_data(30)

# Training Progress Section
st.header("📈 Training Progress")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Loss Over Time")
    chart_data = df[['date', 'train_loss', 'val_loss']].set_index('date')
    st.line_chart(chart_data)

with col2:
    st.subheader("Accuracy Over Time")
    chart_data = df[['date', 'train_acc', 'val_acc']].set_index('date')
    st.line_chart(chart_data)

st.markdown("---")

# Performance Breakdown
st.header("🎯 Performance Breakdown")

# Generate per-class metrics
class_data = pd.DataFrame({
    'Class': ['Class 0', 'Class 1', 'Class 2'],
    'Precision': [0.94, 0.91, 0.89],
    'Recall': [0.92, 0.93, 0.88],
    'F1-Score': [0.93, 0.92, 0.88],
    'Support': [450, 523, 412]
})

st.dataframe(class_data, use_container_width=True)

st.markdown("---")

# Confusion Matrix
st.header("🔢 Confusion Matrix")
confusion_matrix = np.array([
    [420, 18, 12],
    [25, 486, 12],
    [35, 15, 362]
])

st.dataframe(pd.DataFrame(
    confusion_matrix,
    columns=['Predicted 0', 'Predicted 1', 'Predicted 2'],
    index=['True 0', 'True 1', 'True 2']
), use_container_width=True)

st.markdown("---")

# Recent Predictions
st.header("🔍 Recent Predictions")

recent_predictions = pd.DataFrame({
    'Timestamp': [datetime.now() - timedelta(minutes=x) for x in range(10)],
    'Input ID': [f'req_{1000+x}' for x in range(10)],
    'Prediction': np.random.randint(0, 3, 10),
    'Confidence': np.random.uniform(0.7, 0.99, 10),
    'Latency (ms)': np.random.uniform(10, 50, 10)
})

recent_predictions['Confidence'] = recent_predictions['Confidence'].apply(lambda x: f"{x:.2%}")
recent_predictions['Latency (ms)'] = recent_predictions['Latency (ms)'].apply(lambda x: f"{x:.1f}")

st.dataframe(recent_predictions, use_container_width=True)

st.markdown("---")

# System Status
st.header("💻 System Status")

col1, col2, col3 = st.columns(3)

with col1:
    st.success("✅ Model Status: Healthy")
    st.info(f"Model Version: v1.2.3")

with col2:
    st.success("✅ API Status: Running")
    st.info(f"Uptime: 15d 7h 23m")

with col3:
    st.success("✅ Database: Connected")
    st.info(f"Latency: 12ms")

# Footer
st.markdown("---")
st.caption("Dashboard last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
