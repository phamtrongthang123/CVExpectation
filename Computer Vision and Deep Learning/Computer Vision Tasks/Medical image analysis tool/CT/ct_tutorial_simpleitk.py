# /// script
# requires-python = ">=3.8"
# dependencies = [
#     "numpy>=1.21.0",
#     "matplotlib>=3.5.0",
#     "SimpleITK>=2.2.0",
#     "pydicom>=2.3.0",
# ]
# ///
"""
SimpleITK CT Loading Tutorial - Simplified

This script demonstrates the core SimpleITK functionality for loading CT DICOM images.
Focuses on the essential SimpleITK usage while hiding boilerplate code.

Usage: uv run ct_tutorial_simpleitk.py
"""

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from pathlib import Path

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
# CORE SIMPLEITK FUNCTIONALITY: Loading CT Images
print("\nSimpleITK CT Loading - Core Implementation:")
print("=" * 50)

# Step 1: Create SimpleITK ImageSeriesReader
print("Step 1: Creating SimpleITK ImageSeriesReader...")
series_reader = sitk.ImageSeriesReader()

# Step 2: Set the DICOM file names
print("Step 2: Setting DICOM file names...")
series_reader.SetFileNames(dicom_files)

# Step 3: Execute the reader to load the image
print("Step 3: Loading DICOM series...")
ct_image = series_reader.Execute()

# Step 4: Convert SimpleITK image to NumPy array
print("Step 4: Converting to NumPy array...")
ct_array = sitk.GetArrayFromImage(ct_image)

print("Successfully loaded with SimpleITK!")

# Display loaded data info
print("\nLoaded CT Data Summary:")
print(f"   Shape: {ct_array.shape}")
print(f"   Data type: {ct_array.dtype}")
print(f"   Spacing: {ct_image.GetSpacing()} mm")
print(f"   Intensity range: {ct_array.min():.0f} to {ct_array.max():.0f} HU")
print(f"   Number of slices: {ct_array.shape[0]}")

# ================================================
# Simple Visualization
print("\nQuick visualization of loaded CT data...")

# ... [Boilerplate: CT windowing function - hidden for clarity] ...
def apply_ct_windowing(image_array, window_level=40, window_width=400):
    """Hidden function that applies CT windowing for better visualization"""
    window_min = window_level - window_width // 2
    window_max = window_level + window_width // 2
    windowed = np.clip(image_array, window_min, window_max)
    windowed = (windowed - window_min) / (window_max - window_min)
    return windowed

# Simple visualization of middle slice
middle_slice = ct_array[ct_array.shape[0]//2]
windowed_slice = apply_ct_windowing(middle_slice)

plt.figure(figsize=(8, 6))
plt.imshow(windowed_slice, cmap='gray', origin='lower')
plt.title('CT Slice (Loaded with SimpleITK)')
plt.axis('off')
plt.tight_layout()
plt.show()
