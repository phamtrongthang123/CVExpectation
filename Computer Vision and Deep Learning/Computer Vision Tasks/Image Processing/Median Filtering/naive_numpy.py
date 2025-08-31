# /// script
# dependencies = ["numpy", "matplotlib", "pillow"]
# ///

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def median_filter_naive(image, kernel_size=3):
    """Naive median filter implementation using basic numpy operations"""
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    pad = kernel_size // 2
    padded = np.pad(image, pad, mode='reflect')
    filtered = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            window = padded[i:i+kernel_size, j:j+kernel_size]
            filtered[i, j] = np.median(window)
    
    return filtered

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
    noise_ratio = 0.05  # 5% noise
    noise_mask = np.random.rand(*original.shape) < noise_ratio
    
    # Salt noise (white pixels)
    salt_mask = noise_mask & (np.random.rand(*original.shape) > 0.5)
    noisy[salt_mask] = 255
    
    # Pepper noise (black pixels)  
    pepper_mask = noise_mask & ~salt_mask
    noisy[pepper_mask] = 0
    
    print(f"Added {np.sum(noise_mask)} noise pixels ({noise_ratio*100:.1f}%)")
    
    # Apply filter
    print("Applying naive median filter...")
    filtered = median_filter_naive(noisy, kernel_size=5)
    
    # Display results
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(original, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original Lena')
    axes[0].axis('off')
    axes[1].imshow(noisy, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Noisy (Salt & Pepper)')
    axes[1].axis('off')
    axes[2].imshow(filtered, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Filtered (Naive)')
    axes[2].axis('off')
    plt.tight_layout()
    plt.show()
