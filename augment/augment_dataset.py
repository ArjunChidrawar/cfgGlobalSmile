import os
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import random
from pathlib import Path
import argparse

def create_augmented_dataset(input_dir, output_dir, num_augmentations=3):
    """
    Create augmented dataset with various transformations

    Args:
        input_dir: Directory containing original images
        output_dir: Directory to save augmented images
        num_augmentations: Number of augmented versions per image
    """

    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Get all image files
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))

    print(f"Found {len(image_files)} images to augment")

    for img_path in image_files:
        print(f"Processing {img_path.name}...")

        # Load image
        img = Image.open(img_path)

        # Generate multiple augmented versions
        for i in range(num_augmentations):
            # Apply random transformations
            augmented_img = apply_augmentations(img)

            # Save augmented image
            base_name = img_path.stem
            output_name = f"{base_name}_aug_{i+1}.png"
            output_path = Path(output_dir) / output_name

            augmented_img.save(output_path)
            print(f"  Saved: {output_name}")

    print(f"\nAugmentation complete! Check {output_dir} for results.")

def create_augmented_dataset_paired(input_dir_no_mask, input_dir_mask, output_dir_no_mask, output_dir_mask, num_augmentations=3):
    """
    Create augmented dataset for paired images (no-mask and mask), applying identical augmentations to both.
    """
    Path(output_dir_no_mask).mkdir(parents=True, exist_ok=True)
    Path(output_dir_mask).mkdir(parents=True, exist_ok=True)

    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = []
    for ext in image_extensions:
        image_files.extend(Path(input_dir_no_mask).glob(f'*{ext}'))

    print(f"Found {len(image_files)} image pairs to augment")

    for img_path in image_files:
        mask_path = Path(input_dir_mask) / img_path.name
        if not mask_path.exists():
            print(f"Warning: Mask not found for {img_path.name}, skipping.")
            continue
        print(f"Processing {img_path.name}...")
        img = Image.open(img_path)
        mask = Image.open(mask_path)
        for i in range(num_augmentations):
            # Use a fixed random seed for each pair/augmentation to ensure identical transforms
            seed = hash((img_path.name, i)) % (2**32)
            augmented_img = apply_augmentations(img, seed=seed)
            augmented_mask = apply_augmentations(mask, seed=seed)
            base_name = img_path.stem
            output_name = f"{base_name}_aug_{i+1}.png"
            output_path_img = Path(output_dir_no_mask) / output_name
            output_path_mask = Path(output_dir_mask) / output_name
            augmented_img.save(output_path_img)
            augmented_mask.save(output_path_mask)
            print(f"  Saved: {output_name} (image & mask)")
    print(f"\nPaired augmentation complete! Check {output_dir_no_mask} and {output_dir_mask} for results.")

def apply_augmentations(img, seed=None):
    """
    Apply various augmentations to an image. If seed is provided, use it for deterministic augmentations.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # 1. Random rotation (-15 to 15 degrees)
    if random.random() < 0.7:
        angle = random.uniform(-15, 15)
        img = img.rotate(angle, fillcolor=(255, 255, 255))

    # 2. Random horizontal flip
    if random.random() < 0.5:
        img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

    # 3. Random brightness adjustment
    if random.random() < 0.8:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Brightness(img)
        img = enhancer.enhance(factor)

    # 4. Random contrast adjustment
    if random.random() < 0.8:
        factor = random.uniform(0.8, 1.2)
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(factor)

    # 5. Random saturation adjustment
    if random.random() < 0.6:
        factor = random.uniform(0.7, 1.3)
        enhancer = ImageEnhance.Color(img)
        img = enhancer.enhance(factor)

    # 6. Random sharpness adjustment
    if random.random() < 0.5:
        factor = random.uniform(0.5, 1.5)
        enhancer = ImageEnhance.Sharpness(img)
        img = enhancer.enhance(factor)

    # 7. Random blur (slight)
    if random.random() < 0.3:
        radius = random.uniform(0.5, 1.5)
        img = img.filter(ImageFilter.GaussianBlur(radius=radius))

    # 8. Random noise addition
    if random.random() < 0.4:
        img_array = np.array(img)
        noise = np.random.normal(0, random.uniform(5, 15), img_array.shape)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)

    # 9. Random color temperature (warm/cool)
    if random.random() < 0.5:
        img_array = np.array(img)
        # Add slight color tint
        tint_factor = random.uniform(-20, 20)
        if random.random() < 0.5:
            # Warm tint (more red/yellow)
            img_array[:, :, 0] = np.clip(img_array[:, :, 0] + tint_factor, 0, 255)
            img_array[:, :, 1] = np.clip(img_array[:, :, 1] + tint_factor * 0.5, 0, 255)
        else:
            # Cool tint (more blue)
            img_array[:, :, 2] = np.clip(img_array[:, :, 2] + tint_factor, 0, 255)
        img = Image.fromarray(img_array)

    # 10. Random crop and resize (slight)
    if random.random() < 0.4:
        width, height = img.size
        crop_ratio = random.uniform(0.85, 0.95)
        new_width = int(width * crop_ratio)
        new_height = int(height * crop_ratio)

        left = random.randint(0, width - new_width)
        top = random.randint(0, height - new_height)
        right = left + new_width
        bottom = top + new_height

        img = img.crop((left, top, right, bottom))
        img = img.resize((width, height), Image.Resampling.LANCZOS)

    return img

def create_specific_augmentations(input_dir, output_dir):
    """
    Create specific types of augmentations for analysis
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    image_files = list(Path(input_dir).glob('*.png'))

    for img_path in image_files[:5]:  # Test with first 5 images
        print(f"Creating specific augmentations for {img_path.name}...")

        img = Image.open(img_path)
        if img.mode != 'RGB':
            img = img.convert('RGB')

        base_name = img_path.stem

        # 1. Bright variations
        for i, factor in enumerate([0.6, 0.8, 1.2, 1.4]):
            bright_img = ImageEnhance.Brightness(img).enhance(factor)
            bright_img.save(Path(output_dir) / f"{base_name}_bright_{i+1}.png")

        # 2. Contrast variations
        for i, factor in enumerate([0.6, 0.8, 1.2, 1.4]):
            contrast_img = ImageEnhance.Contrast(img).enhance(factor)
            contrast_img.save(Path(output_dir) / f"{base_name}_contrast_{i+1}.png")

        # 3. Rotation variations
        for i, angle in enumerate([-10, -5, 5, 10]):
            rotated_img = img.rotate(angle, fillcolor=(255, 255, 255))
            rotated_img.save(Path(output_dir) / f"{base_name}_rotate_{i+1}.png")

        # 4. Flipped version
        flipped_img = img.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        flipped_img.save(Path(output_dir) / f"{base_name}_flipped.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Augment dataset with various transformations')
    parser.add_argument('--input_dir', default='results/inpaint/masked',
                       help='Input directory containing original images')
    parser.add_argument('--output_dir', default='augmented_dataset',
                       help='Output directory for augmented images')
    parser.add_argument('--num_augmentations', type=int, default=3,
                       help='Number of augmented versions per image')
    parser.add_argument('--specific', action='store_true',
                       help='Create specific augmentations for analysis')
    # New arguments for paired mode
    parser.add_argument('--input_dir_no_mask', type=str, default=None, help='Input directory for no-mask images')
    parser.add_argument('--input_dir_mask', type=str, default=None, help='Input directory for mask images')
    parser.add_argument('--output_dir_no_mask', type=str, default=None, help='Output directory for augmented no-mask images')
    parser.add_argument('--output_dir_mask', type=str, default=None, help='Output directory for augmented mask images')

    args = parser.parse_args()

    if args.input_dir_no_mask and args.input_dir_mask and args.output_dir_no_mask and args.output_dir_mask:
        create_augmented_dataset_paired(
            args.input_dir_no_mask,
            args.input_dir_mask,
            args.output_dir_no_mask,
            args.output_dir_mask,
            args.num_augmentations
        )
    elif args.specific:
        create_specific_augmentations(args.input_dir, args.output_dir)
    else:
        create_augmented_dataset(args.input_dir, args.output_dir, args.num_augmentations)
