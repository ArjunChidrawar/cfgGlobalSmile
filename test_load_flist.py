#!/usr/bin/env python3

import os
import numpy as np
import glob

def load_flist(flist):
    if isinstance(flist, list):
        return flist

    # flist: image file path, image directory path, text file flist path
    if isinstance(flist, str):
        if os.path.isdir(flist):
            flist = list(glob.glob(flist + '/*.jpg')) + list(glob.glob(flist + '/*.png'))
            flist.sort()
            return flist

        if os.path.isfile(flist):
            try:
                result = np.genfromtxt(flist, dtype=str, encoding='utf-8')
                print(f"Loaded {len(result)} items from {flist}")
                print(f"First few items: {result[:3]}")
                return result
            except Exception as e:
                print(f"Error loading {flist}: {e}")
                return [flist]

    return []

# Test the function
print("Testing load_flist with input_flist.txt...")
result = load_flist("input_flist.txt")
print(f"Result type: {type(result)}")
print(f"Result length: {len(result)}")
if len(result) > 0:
    print(f"First item: {result[0]}")
    print(f"Last item: {result[-1]}")

    # Test if the first item exists
    first_path = result[0]
    print(f"First path: {first_path}")
    print(f"File exists: {os.path.exists(first_path)}")

    # Test if we can read it as an image
    try:
        from imageio import imread
        img = imread(first_path)
        print(f"Successfully read image: {img.shape}")
    except Exception as e:
        print(f"Error reading image: {e}")
