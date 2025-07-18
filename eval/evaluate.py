import os
import glob
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from skimage.io import imread
import numpy as np

# Directories
outputs_dir = 'results/inpaint/result'
postop_dir = 'augment/augmented_no_mask'

# Output CSV
csv_path = 'outputs_2/evaluation_results.csv'

# Helper to get base name without extension
get_base = lambda path: os.path.splitext(os.path.basename(path))[0]

# Gather output images
output_images = sorted(glob.glob(os.path.join(outputs_dir, '*.png')) + glob.glob(os.path.join(outputs_dir, '*.jpg')))

results = []

for out_path in output_images:
    base = get_base(out_path)
    # Try to find the corresponding post-op image
    postop_path = os.path.join(postop_dir, f"{base}.png")

    if os.path.exists(postop_path):
        try:
            # Load images
            output_img = imread(out_path)
            postop_img = imread(postop_path)

            # Ensure same size (resize post-op to match output if needed)
            if output_img.shape != postop_img.shape:
                from skimage.transform import resize
                postop_img = resize(postop_img, output_img.shape, preserve_range=True).astype(postop_img.dtype)

            # Convert to float for metrics
            output_img = output_img.astype(np.float64) / 255.0
            postop_img = postop_img.astype(np.float64) / 255.0

            # Calculate metrics
            psnr_val = psnr(postop_img, output_img, data_range=1.0)

            # SSIM with proper channel handling
            if len(output_img.shape) == 3 and output_img.shape[2] == 3:
                ssim_val = ssim(postop_img, output_img, channel_axis=2, data_range=1.0)
            else:
                ssim_val = ssim(postop_img, output_img, data_range=1.0)

            mae_val = np.mean(np.abs(postop_img - output_img))

            results.append({
                'Image': base,
                'PSNR': psnr_val,
                'SSIM': ssim_val,
                'MAE': mae_val
            })

            print(f"{base}: PSNR={psnr_val:.3f}, SSIM={ssim_val:.3f}, MAE={mae_val:.3f}")

        except Exception as e:
            print(f"Error processing {base}: {e}")
    else:
        print(f"No post-op image found for {base}")

# Save results to CSV
if results:
    # Ensure output directory exists
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)

    with open(csv_path, 'w', newline='') as csvfile:
        fieldnames = ['Image', 'PSNR', 'SSIM', 'MAE']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    # Calculate comprehensive statistics
    psnr_values = [r['PSNR'] for r in results]
    ssim_values = [r['SSIM'] for r in results]
    mae_values = [r['MAE'] for r in results]

    stats = {
        'PSNR': {
            'Mean': np.mean(psnr_values),
            'Median': np.median(psnr_values),
            'Std': np.std(psnr_values),
            'Min': np.min(psnr_values),
            'Max': np.max(psnr_values),
            'Range': np.max(psnr_values) - np.min(psnr_values),
            'Q1': np.percentile(psnr_values, 25),
            'Q3': np.percentile(psnr_values, 75),
            'IQR': np.percentile(psnr_values, 75) - np.percentile(psnr_values, 25)
        },
        'SSIM': {
            'Mean': np.mean(ssim_values),
            'Median': np.median(ssim_values),
            'Std': np.std(ssim_values),
            'Min': np.min(ssim_values),
            'Max': np.max(ssim_values),
            'Range': np.max(ssim_values) - np.min(ssim_values),
            'Q1': np.percentile(ssim_values, 25),
            'Q3': np.percentile(ssim_values, 75),
            'IQR': np.percentile(ssim_values, 75) - np.percentile(ssim_values, 25)
        },
        'MAE': {
            'Mean': np.mean(mae_values),
            'Median': np.median(mae_values),
            'Std': np.std(mae_values),
            'Min': np.min(mae_values),
            'Max': np.max(mae_values),
            'Range': np.max(mae_values) - np.min(mae_values),
            'Q1': np.percentile(mae_values, 25),
            'Q3': np.percentile(mae_values, 75),
            'IQR': np.percentile(mae_values, 75) - np.percentile(mae_values, 25)
        }
    }

    # Save statistics to a separate CSV (append if exists)
    stats_csv_path = 'outputs_2/evaluation_statistics.csv'
    os.makedirs(os.path.dirname(stats_csv_path), exist_ok=True)
    file_exists = os.path.isfile(stats_csv_path)

    with open(stats_csv_path, 'a', newline='') as csvfile:
        fieldnames = ['Metric', 'Mean', 'Median', 'Std', 'Min', 'Max', 'Range', 'Q1', 'Q3', 'IQR']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for metric, values in stats.items():
            writer.writerow({'Metric': metric, **values})

    print(f"\n=== COMPREHENSIVE STATISTICS ===")
    for metric, values in stats.items():
        print(f"\n{metric}:")
        for stat, value in values.items():
            print(f"  {stat}: {value:.4f}")

    print(f"\nResults saved to {csv_path}")
    print(f"Statistics saved to {stats_csv_path}")
else:
    print("No matching pairs found!")
