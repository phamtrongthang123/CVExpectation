# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy>=1.21.0",
#     "matplotlib>=3.5.0",
#     "monai>=1.3.0",
#     "pydicom>=2.3.0",
#     "torch>=1.12.0",
#     "simpleitk>=2.5.0",
# ]
# ///
"""
MONAI CT Loading Tutorial - Simplified

This script demonstrates the core MONAI functionality for loading CT DICOM images.
Focuses on the essential MONAI usage while hiding boilerplate code.

Usage: uv run ct_tutorial_monai.py
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path

# Core MONAI imports for medical image loading
from monai.data import ITKReader
from monai.transforms import LoadImage

# ... [Boilerplate: DICOM file discovery and sorting - hidden for clarity] ...
def get_prepared_dicom_files():
    """Hidden boilerplate function that handles DICOM file discovery and sorting"""
    import pydicom
    
    dicom_dir = Path("data_example")
    dcm_files = list(dicom_dir.glob("*.dcm"))
    
    # Group by Series Number and select Series 5
    series_dict = {}
    for dcm_file in dcm_files:
        dcm = pydicom.dcmread(str(dcm_file), stop_before_pixels=True)
        series_num = int(dcm.SeriesNumber)
        if series_num not in series_dict:
            series_dict[series_num] = []
        series_dict[series_num].append(str(dcm_file))
    
    dicom_files = series_dict[5]  # Select Series 5
    
    # Sort by Instance Number
    def get_instance_number(dcm_file):
        dcm = pydicom.dcmread(dcm_file, stop_before_pixels=True)
        return int(dcm.InstanceNumber) if hasattr(dcm, 'InstanceNumber') else 0
    
    dicom_files.sort(key=get_instance_number)
    return dicom_files

# Get prepared DICOM files
dicom_files = get_prepared_dicom_files()
print(f"Found {len(dicom_files)} DICOM files to load")

# ================================================
# CORE MONAI FUNCTIONALITY: Loading CT Images
print("\nMONAI CT Loading - Core Implementation:")
print("=" * 50)

# Step 1: Create MONAI loader with ITKReader
print("Step 1: Creating MONAI loader...")
loader = LoadImage(reader=ITKReader(), dtype=np.float32)

# Step 2: Load DICOM series with MONAI
print("Step 2: Loading DICOM series with MONAI...")
try:
    ct_data = loader(dicom_files)  # Pass list of DICOM files to MONAI
    
    # Step 3: Extract data from MONAI loader result
    print("Step 3: Extracting data from MONAI loader...")
    ct_array = ct_data[0]        # Image data (first element)
    metadata = ct_data[1]        # Metadata (second element)
    
    # Convert from PyTorch tensor to NumPy (if needed)
    if torch.is_tensor(ct_array):
        ct_array = ct_array.numpy()
    
    print("Successfully loaded with MONAI ITKReader!")
    
except Exception as e:
    print(f"MONAI ITKReader failed: {e}")
    print("Falling back to SimpleITK...")
    
    # Fallback using SimpleITK (non-MONAI approach), because MONAI will assume that all spacing is the same, which is not always true in practice.
    import SimpleITK as sitk
    series_reader = sitk.ImageSeriesReader()
    series_reader.SetFileNames(dicom_files)
    sitk_image = series_reader.Execute()
    ct_array = sitk.GetArrayFromImage(sitk_image).astype(np.float32)

# Display loaded data info
print("\nLoaded CT Data Summary:")
print(f"   Shape: {ct_array.shape}")
print(f"   Data type: {ct_array.dtype}")
print(f"   Intensity range: {ct_array.min():.0f} to {ct_array.max():.0f} HU")
print(f"   Number of slices: {ct_array.shape[0]}")

# ================================================
# Simple Visualization 
print("\nQuick visualization of loaded CT data...")

def apply_ct_windowing(image_array, window_level=40, window_width=400):
    """Hidden function that applies CT windowing for better visualization"""
    window_min = window_level - window_width // 2
    window_max = window_level + window_width // 2
    windowed = np.clip(image_array, window_min, window_max)
    windowed = (windowed - window_min) / (window_max - window_min)
    return windowed

# Handle MONAI array dimensions (remove singleton dims if present)
if len(ct_array.shape) == 4:
    ct_array = np.squeeze(ct_array)

# Simple visualization of middle slice
middle_slice = ct_array[ct_array.shape[0]//2]
windowed_slice = apply_ct_windowing(middle_slice)

plt.figure(figsize=(8, 6))
plt.imshow(windowed_slice, cmap='gray', origin='lower')
plt.title('CT Slice (Loaded with MONAI)')
plt.axis('off')
plt.tight_layout()
plt.show()
