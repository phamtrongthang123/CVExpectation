#!/usr/bin/env uv run
# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "opencv-python",
#     "torch",
#     "torchvision",
#     "numpy",
# ]
# ///

"""
Simple real-time video analysis tool using OpenCV.
Demonstrates: video capture, frame processing, and real-time inference.

Run with: uv run simple_video_analysis.py
Press 'q' to quit
"""

import cv2
import torch
import torchvision.transforms as transforms
from torchvision import models
import numpy as np

# Load pretrained model
print("Loading model...")
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Simplified ImageNet classes
CLASSES = ['background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane']

# Image preprocessing
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def process_frame(frame):
    """Process a single video frame"""
    # Preprocess
    input_tensor = transform(frame)
    input_batch = input_tensor.unsqueeze(0)

    # Inference
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get top prediction
    top_prob, top_class = probabilities.topk(1)

    return top_class.item(), top_prob.item()

def add_text_to_frame(frame, text, position=(10, 30)):
    """Add text overlay to frame"""
    # Add background for text
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(frame, (position[0] - 5, position[1] - text_height - 5),
                  (position[0] + text_width + 5, position[1] + 5), (0, 0, 0), -1)

    # Add text
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

def main():
    """Main video analysis loop"""
    print("Starting video analysis...")
    print("Press 'q' to quit")

    # Open webcam (0 is default camera)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam")
        print("Running in demo mode with synthetic frames...")

        # Demo mode: process synthetic frames
        for i in range(100):
            # Create synthetic frame
            frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

            # Add text
            cv2.putText(frame, "Demo Mode - No Camera", (50, 240),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Frame {i}/100", (50, 280),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            print(f"Processing frame {i}/100")

            if i >= 99:
                break

        print("Demo completed!")
        return

    # Real-time processing
    frame_count = 0
    fps_list = []

    while True:
        # Capture frame
        ret, frame = cap.read()

        if not ret:
            print("Error: Failed to capture frame")
            break

        start_time = cv2.getTickCount()

        # Process every 5th frame to reduce computational load
        if frame_count % 5 == 0:
            try:
                # Classify frame
                class_id, confidence = process_frame(frame)

                # Display results
                class_name = CLASSES[class_id] if class_id < len(CLASSES) else f"class_{class_id}"
                text = f"{class_name}: {confidence*100:.1f}%"

            except Exception as e:
                text = f"Error: {str(e)[:30]}"

        # Calculate FPS
        end_time = cv2.getTickCount()
        time_taken = (end_time - start_time) / cv2.getTickFrequency()
        fps = 1 / time_taken if time_taken > 0 else 0
        fps_list.append(fps)

        # Keep only last 30 FPS measurements
        if len(fps_list) > 30:
            fps_list.pop(0)

        avg_fps = np.mean(fps_list)

        # Add overlays
        add_text_to_frame(frame, text, (10, 30))
        add_text_to_frame(frame, f"FPS: {avg_fps:.1f}", (10, 70))
        add_text_to_frame(frame, f"Frame: {frame_count}", (10, 110))

        # Display
        cv2.imshow('Real-time Video Analysis', frame)

        frame_count += 1

        # Check for quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()

    print(f"\nProcessed {frame_count} frames")
    print(f"Average FPS: {np.mean(fps_list):.2f}")

if __name__ == "__main__":
    main()
