#!/usr/bin/env python3

import os
import numpy as np
import sys
sys.path.append('src')

# Import with absolute paths to avoid relative import issues
import importlib.util
spec = importlib.util.spec_from_file_location("dataset", "src/dataset.py")
dataset_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dataset_module)

spec = importlib.util.spec_from_file_location("config", "src/config.py")
config_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_module)

def test_dataset_loading():
    # Load config
    config = config_module.Config('checkpoints/config.yml')

    print("Config loaded successfully")
    print(f"TRAIN_INPAINT_IMAGE_FLIST: {config.TRAIN_INPAINT_IMAGE_FLIST}")
    print(f"TRAIN_INPAINT_LANDMARK_FLIST: {config.TRAIN_INPAINT_LANDMARK_FLIST}")
    print(f"TRAIN_MASK_FLIST: {config.TRAIN_MASK_FLIST}")

    # Test load_flist method directly
    dataset = dataset_module.Dataset(config, config.TRAIN_INPAINT_IMAGE_FLIST,
                     config.TRAIN_INPAINT_LANDMARK_FLIST,
                     config.TRAIN_MASK_FLIST, augment=True, training=True)

    print(f"\nDataset length: {len(dataset)}")
    print(f"Dataset data length: {len(dataset.data)}")
    print(f"Dataset landmark_data length: {len(dataset.landmark_data)}")
    print(f"Dataset mask_data length: {len(dataset.mask_data)}")

    # Print first few entries
    print("\nFirst 3 image paths:")
    for i in range(min(3, len(dataset.data))):
        print(f"  {i}: {dataset.data[i]}")

    print("\nFirst 3 landmark paths:")
    for i in range(min(3, len(dataset.landmark_data))):
        print(f"  {i}: {dataset.landmark_data[i]}")

    print("\nFirst 3 mask paths:")
    for i in range(min(3, len(dataset.mask_data))):
        print(f"  {i}: {dataset.mask_data[i]}")

    # Test loading first item
    try:
        print("\nTrying to load first item...")
        item = dataset[0]
        print("Successfully loaded first item!")
    except Exception as e:
        print(f"Error loading first item: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dataset_loading()
