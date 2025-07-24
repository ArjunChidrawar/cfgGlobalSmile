import os
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--paths', nargs='+', type=str, default=[
    'augment/augmented_no_mask',
    'augmented_cleft_landmarks_text',
    'augment/augmented_mask'], help='List of input directories')
parser.add_argument('--outputs', nargs='+', type=str, default=[
    'augmented_no_mask_flist.txt',
    'augmented_landmarks_flist.txt',
    'augmented_mask_flist.txt'], help='List of output flist files')
args = parser.parse_args()

ext = {'.jpg', '.png', '.txt'}

for in_path, out_file in zip(args.paths, args.outputs):
    images = []
    for root, dirs, files in os.walk(in_path):
        print('loading ' + root)
        for file in files:
            if os.path.splitext(file)[1] in ext:
                images.append(os.path.join(root, file))
    images = sorted(images)
    np.savetxt(out_file, images, fmt='%s')
