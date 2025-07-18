import os
import glob
from skimage.io import imread
import matplotlib.pyplot as plt

# Directories
outputs_dir = 'results/inpaint/result'
postop_dir = 'augment/augmented_no_mask'
save_dir = 'outputs_2/side_by_side_comparisons'
os.makedirs(save_dir, exist_ok=True)

# Gather output images
output_images = sorted(glob.glob(os.path.join(outputs_dir, '*.png')) + glob.glob(os.path.join(outputs_dir, '*.jpg')))

for out_path in output_images:
    base = os.path.splitext(os.path.basename(out_path))[0]
    postop_path = os.path.join(postop_dir, f"{base}.png")
    if os.path.exists(postop_path):
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
    else:
        print(f'No post-op image found for {base}')
