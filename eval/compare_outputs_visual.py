import os
import glob
from skimage.io import imread
import matplotlib.pyplot as plt

# Directories
outputs_dir = 'results/inpaint/result'
postop_dir = 'augment/augmented_no_mask'
save_dir = 'outputs_3/side_by_side_comparisons'
os.makedirs(save_dir, exist_ok=True)

# Flist path
flist_path = 'output_flist/val_images.flist'

# Read image basenames from flist
with open(flist_path, 'r') as f:
    image_basenames = [os.path.splitext(os.path.basename(line.strip()))[0] for line in f if line.strip()]

for base in image_basenames:
    # Try both .png and .jpg for output image
    out_path = os.path.join(outputs_dir, f"{base}.png")
    if not os.path.exists(out_path):
        out_path = os.path.join(outputs_dir, f"{base}.jpg")
        if not os.path.exists(out_path):
            print(f"No output image found for {base}")
            continue
    # Try both .png and .jpg for postop image
    postop_path = os.path.join(postop_dir, f"{base}.png")
    if not os.path.exists(postop_path):
        postop_path = os.path.join(postop_dir, f"{base}.jpg")
        if not os.path.exists(postop_path):
            print(f"No post-op image found for {base}")
            continue
    output_img = imread(out_path)
    postop_img = imread(postop_path)
    # Resize post-op to match output if needed
    if output_img.shape != postop_img.shape:
        from skimage.transform import resize
        postop_img = resize(postop_img, output_img.shape, preserve_range=True).astype(output_img.dtype)
    # Create side by side figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(output_img)
    axes[0].set_title('Model Output')
    axes[0].axis('off')
    axes[1].imshow(postop_img)
    axes[1].set_title('Real Post-Op')
    axes[1].axis('off')
    plt.suptitle(base)
    plt.tight_layout()
    # Save to file
    save_path = os.path.join(save_dir, f'{base}_side_by_side.png')
    plt.savefig(save_path)
    plt.close(fig)
    print(f'Saved: {save_path}')
