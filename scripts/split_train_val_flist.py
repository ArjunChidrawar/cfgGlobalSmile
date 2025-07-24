import argparse
import random
import os

parser = argparse.ArgumentParser(description="Split a training flist into new train/val flists with corresponding masks and landmarks.")
parser.add_argument('--train_image_flist', type=str, default='augmented_no_mask_flist.txt', help='Path to the training image flist (default: train_images.flist)')
parser.add_argument('--train_mask_flist', type=str, default='augmented_mask_flist.txt', help='Path to the training mask flist (default: train_masks.flist)')
parser.add_argument('--train_landmark_flist', type=str, default='augmented_landmarks_flist.txt', help='Path to the training landmark flist (default: train_landmarks.flist)')
parser.add_argument('--val_size', type=int, default=None, help='Number of samples to use for validation (overrides val_frac, default: None)')
parser.add_argument('--val_frac', type=float, default=0.15, help='Fraction of samples to use for validation if val_size not set (default: 0.1)')
parser.add_argument('--output_dir', type=str, default='output_flist', help='Directory to save new flists (default: output_flist)')

args = parser.parse_args()

# Read all flists
with open(args.train_image_flist, 'r') as f:
    images = [line.strip() for line in f if line.strip()]
with open(args.train_mask_flist, 'r') as f:
    masks = [line.strip() for line in f if line.strip()]
with open(args.train_landmark_flist, 'r') as f:
    landmarks = [line.strip() for line in f if line.strip()]

assert len(images) == len(masks) == len(landmarks), "All flists must have the same number of entries."

num_samples = len(images)
if args.val_size is not None:
    val_size = args.val_size
else:
    val_size = int(num_samples * args.val_frac)
val_size = min(val_size, num_samples)

indices = list(range(num_samples))
random.shuffle(indices)
val_indices = set(indices[:val_size])
train_indices = set(indices[val_size:])

train_images = [images[i] for i in train_indices]
train_masks = [masks[i] for i in train_indices]
train_landmarks = [landmarks[i] for i in train_indices]

val_images = [images[i] for i in val_indices]
val_masks = [masks[i] for i in val_indices]
val_landmarks = [landmarks[i] for i in val_indices]

os.makedirs(args.output_dir, exist_ok=True)

def write_flist(lst, path):
    with open(path, 'w') as f:
        for item in lst:
            f.write(item + '\n')

write_flist(train_images, os.path.join(args.output_dir, 'train_images.flist'))
write_flist(train_masks, os.path.join(args.output_dir, 'train_masks.flist'))
write_flist(train_landmarks, os.path.join(args.output_dir, 'train_landmarks.flist'))

write_flist(val_images, os.path.join(args.output_dir, 'val_images.flist'))
write_flist(val_masks, os.path.join(args.output_dir, 'val_masks.flist'))
write_flist(val_landmarks, os.path.join(args.output_dir, 'val_landmarks.flist'))

print(f"Split complete. {len(train_images)} train, {len(val_images)} val samples written to {args.output_dir}")
