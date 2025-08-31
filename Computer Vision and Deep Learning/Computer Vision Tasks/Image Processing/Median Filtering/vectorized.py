# /// script
# dependencies = ["numpy", "matplotlib", "pillow"]
# ///

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def median_filter_vectorized(image, kernel_size=3):
    """Vectorized median filter using numpy operations"""
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    
    # Create sliding windows using strides
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    windows = windows.reshape(h, w, -1)
    
    return np.median(windows, axis=2)

def median_filter_optimized(image, kernel_size=3):
    """Optimized vectorized approach using array operations"""
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    h, w = image.shape
    
    # Vectorized median using percentile
    windows = np.lib.stride_tricks.sliding_window_view(padded, (kernel_size, kernel_size))
    return np.percentile(windows.reshape(h, w, -1), 50, axis=2)

if __name__ == "__main__":
    # Load Lena image
    try:
        original = np.array(Image.open('lena.jpg').convert('L'), dtype=np.float32)
        print(f"Loaded lena.jpg: {original.shape}")
    except FileNotFoundError:
        print("lena.jpg not found, creating synthetic image")
        original = np.random.rand(512, 512) * 255
    
    # Add salt and pepper noise
    noisy = original.copy()
    noise_ratio = 0.08  # 8% noise
    noise_mask = np.random.rand(*original.shape) < noise_ratio
    
    # Salt noise (white pixels)
    salt_mask = noise_mask & (np.random.rand(*original.shape) > 0.5)
    noisy[salt_mask] = 255
    
    # Pepper noise (black pixels)  
    pepper_mask = noise_mask & ~salt_mask
    noisy[pepper_mask] = 0
    
    print(f"Added {np.sum(noise_mask)} noise pixels ({noise_ratio*100:.1f}%)")
    
    # Apply filters
    print("Applying vectorized median filter...")
    filtered_vectorized = median_filter_vectorized(noisy, kernel_size=5)
    
    print("Applying optimized median filter...")
    filtered_optimized = median_filter_optimized(noisy, kernel_size=5)
    
    # Display results
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    axes[0,0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0,0].set_title('Original Lena')
    axes[0,0].axis('off')
    axes[0,1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    axes[0,1].set_title('Noisy (Salt & Pepper)')
    axes[0,1].axis('off')
    axes[1,0].imshow(filtered_vectorized, cmap='gray', vmin=0, vmax=255)
    axes[1,0].set_title('Vectorized Numpy')
    axes[1,0].axis('off')
    axes[1,1].imshow(filtered_optimized, cmap='gray', vmin=0, vmax=255)
    axes[1,1].set_title('Optimized Numpy')
    axes[1,1].axis('off')
    plt.tight_layout()
    plt.show()
