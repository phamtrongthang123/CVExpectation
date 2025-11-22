#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "flask",
#     "torch",
#     "numpy",
# ]
# ///

"""
Simple REST API for serving ML models using Flask.
Demonstrates: model serving, API endpoints, and request handling.

Run with: uv run simple_ml_api.py
Test with: curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d '{"features": [1,2,3,4,5,6,7,8,9,10]}'
"""

from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import numpy as np

# Simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return self.network(x)

# Initialize Flask app
app = Flask(__name__)

# Load model (in production, load from saved checkpoint)
model = SimpleModel()
model.eval()

# Global stats
stats = {
    'total_requests': 0,
    'successful_predictions': 0,
    'failed_predictions': 0
}

@app.route('/')
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'ML Model API',
        'version': '1.0',
        'endpoints': {
            '/': 'API information',
            '/health': 'Health check',
            '/predict': 'Make prediction (POST)',
            '/stats': 'API statistics'
        }
    })

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Prediction endpoint"""
    global stats
    stats['total_requests'] += 1

    try:
        # Parse request
        data = request.get_json()

        if 'features' not in data:
            stats['failed_predictions'] += 1
            return jsonify({'error': 'Missing "features" field'}), 400

        features = data['features']

        # Validate input
        if len(features) != 10:
            stats['failed_predictions'] += 1
            return jsonify({'error': 'Expected 10 features'}), 400

        # Convert to tensor
        input_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Make prediction
        with torch.no_grad():
            output = model(input_tensor)
            probabilities = torch.softmax(output, dim=1)
            prediction = output.argmax(dim=1).item()
            confidence = probabilities[0][prediction].item()

        stats['successful_predictions'] += 1

        # Return result
        return jsonify({
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probabilities[0].tolist()
        })

    except Exception as e:
        stats['failed_predictions'] += 1
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()

        if 'batch' not in data:
            return jsonify({'error': 'Missing "batch" field'}), 400

        batch = data['batch']

        # Convert to tensor
        input_tensor = torch.tensor(batch, dtype=torch.float32)

        # Make predictions
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predictions = outputs.argmax(dim=1).tolist()
            confidences = [probabilities[i][pred].item() for i, pred in enumerate(predictions)]

        return jsonify({
            'predictions': predictions,
            'confidences': confidences,
            'count': len(predictions)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/stats')
def get_stats():
    """API statistics endpoint"""
    return jsonify(stats)

if __name__ == '__main__':
    print("Starting ML Model API Server...")
    print("Endpoints:")
    print("  GET  / - API information")
    print("  GET  /health - Health check")
    print("  POST /predict - Single prediction")
    print("  POST /batch_predict - Batch prediction")
    print("  GET  /stats - API statistics")
    print("\nExample request:")
    print('  curl -X POST http://localhost:5000/predict -H "Content-Type: application/json" -d \'{"features": [1,2,3,4,5,6,7,8,9,10]}\'')
    print("\nStarting server on http://localhost:5000")

    app.run(debug=True, host='0.0.0.0', port=5000)
