#!/usr/bin/env python3

import numpy as np
from imageio import imread
from skimage.color import rgb2gray
from PIL import Image

def analyze_mask_processing():
    # Test with a few mask files
    mask_files = [
        'Binary_mask_resized/Patient_001_2.jpg',
        'Binary_mask_resized/Patient_001_3.jpg',
        'Binary_mask_resized/Patient_001_4.jpg',
        'Binary_mask_resized/Patient_001_5.jpg',
        'Binary_mask_resized/Patient_001_6.jpg'
    ]

    print("Analyzing mask processing pipeline...")
    print("=" * 50)

    all_gray_values = []

    for mask_file in mask_files:
        print(f"\nProcessing: {mask_file}")

        # Step 1: Load original mask
        mask = imread(mask_file)
        print(f"  Original shape: {mask.shape}")
        print(f"  Original range: [{mask.min()}, {mask.max()}]")
        print(f"  Original unique values (first 10): {np.unique(mask)[:10]}")

        # Step 2: Resize (simulate the resize operation)
        mask_resized = np.array(Image.fromarray(mask).resize((256, 256)))
        print(f"  After resize range: [{mask_resized.min()}, {mask_resized.max()}]")

        # Step 3: Convert to grayscale
        mask_gray = rgb2gray(mask_resized)
        print(f"  After rgb2gray range: [{mask_gray.min():.3f}, {mask_gray.max():.3f}]")
        print(f"  After rgb2gray unique values (first 10): {np.unique(mask_gray)[:10]}")

        # Collect all grayscale values for analysis
        all_gray_values.extend(mask_gray.flatten())

        # Step 4: Test different thresholds
        thresholds = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        print("  Testing thresholds:")
        for thresh in thresholds:
            mask_thresholded = (mask_gray > thresh).astype(np.uint8) * 255
            non_zero_pixels = np.sum(mask_thresholded > 0)
            print(f"    threshold {thresh}: {non_zero_pixels} non-zero pixels")

    # Overall statistics
    all_gray_values = np.array(all_gray_values)
    print(f"\n" + "=" * 50)
    print("OVERALL STATISTICS:")
    print(f"All grayscale values range: [{all_gray_values.min():.3f}, {all_gray_values.max():.3f}]")
    print(f"Mean grayscale value: {all_gray_values.mean():.3f}")
    print(f"Median grayscale value: {np.median(all_gray_values):.3f}")
    print(f"Standard deviation: {all_gray_values.std():.3f}")

    # Percentiles
    percentiles = [10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentiles:")
    for p in percentiles:
        value = np.percentile(all_gray_values, p)
        print(f"  {p}th percentile: {value:.3f}")

    # Recommend threshold
    print(f"\nRECOMMENDATION:")
    print(f"Since grayscale values range from {all_gray_values.min():.3f} to {all_gray_values.max():.3f}")
    print(f"A good threshold would be around {all_gray_values.mean():.3f} or lower")
    print(f"Try threshold = 0.1 or 0.2")

if __name__ == "__main__":
    analyze_mask_processing()
