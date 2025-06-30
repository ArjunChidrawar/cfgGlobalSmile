#!/usr/bin/env python3

import os
import argparse
import shutil
import tempfile
import subprocess
import yaml

from PIL import Image
import numpy as np
import cv2
import face_alignment
from skimage import io

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate landmarks+mask, patch config and run test.py"
    )
    p.add_argument(
        "--folder",
        required=True,
        help="Directory containing the uploaded images"
    )
    p.add_argument(
        "--photo",
        required=True,
        help="Filename of the source photo (e.g. 'photo_01.jpg')"
    )
    p.add_argument(
        "--mask",
        required=True,
        help="Filename of the painted‐white mask (e.g. 'masked_01.jpg')"
    )
    return p.parse_args()


def convert_pngs_to_jpgs(folder):
    """Convert any .png in folder to .jpg (delete .png afterward)."""
    for fname in os.listdir(folder):
        if not fname.lower().endswith(".png"):
            continue

        png_path = os.path.join(folder, fname)
        jpg_name = os.path.splitext(fname)[0] + ".jpg"
        jpg_path = os.path.join(folder, jpg_name)

        with Image.open(png_path) as im:
            rgb = im.convert("RGB")
            rgb.save(jpg_path, quality=95)

        os.remove(png_path)


def generate_landmarks(folder, photo):
    """Run face_alignment, draw landmarks and save .txt coords."""
    fa = face_alignment.FaceAlignment(
        face_alignment.LandmarksType.TWO_D, flip_input=False, device='cpu'
    )
    img_path = os.path.join(folder, photo)
    img = io.imread(img_path)
    preds = fa.get_landmarks(img)
    if not preds:
        raise RuntimeError(f"No faces detected in {photo}")

    lm = preds[0]  # (68,2)
    # draw and save image with landmarks
    img_lm = img.copy()
    for x, y in lm:
        cv2.circle(img_lm, (int(x), int(y)), 2, (0,255,0), -1)
    lm_jpg = photo.replace('.jpg','_landmark.jpg')
    cv2.imwrite(os.path.join(folder, lm_jpg), cv2.cvtColor(img_lm, cv2.COLOR_RGB2BGR))

    # save flattened coords to txt
    flat = lm.flatten()
    lm_txt = photo.replace('.jpg','_landmark.txt')
    with open(os.path.join(folder, lm_txt), 'w') as f:
        f.write(" ".join(map(str,flat)) + "\n")

    return lm_jpg, lm_txt


def generate_binary_mask(folder, mask):
    """Threshold the user-painted mask into a strict 0/255 binary image."""
    mask_path = os.path.join(folder, mask)
    img = cv2.imread(mask_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bin_mask = cv2.threshold(gray, 252, 255, cv2.THRESH_BINARY)
    outname = 'binary_mask.jpg'
    cv2.imwrite(os.path.join(folder, outname), bin_mask)
    return outname


def main():
    args = parse_args()

    # 1) ensure jpgs
    convert_pngs_to_jpgs(args.folder)

    # 2) landmarks
    lm_jpg, lm_txt = generate_landmarks(args.folder, args.photo)
    print("Landmarks saved:", lm_jpg, lm_txt)

    # 3) binary mask
    bin_mask = generate_binary_mask(args.folder, args.mask)
    print("Binary mask saved:", bin_mask)

    # 4) patch config & run test
    CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'checkpoints', 'config.yml')

    def make_flist(tmp_dir, name, src_dir, filename):
        p = os.path.join(tmp_dir, name)
        with open(p,'w') as f:
            f.write(os.path.join(src_dir, filename) + '\n')
        return p

    with tempfile.TemporaryDirectory() as tmp:
        # temp subdirs
        photos_dir    = os.path.join(tmp,'photos');    os.makedirs(photos_dir)
        masks_dir     = os.path.join(tmp,'masks');     os.makedirs(masks_dir)
        landmarks_dir = os.path.join(tmp,'landmarks'); os.makedirs(landmarks_dir)

        # copy the three files into temp
        shutil.copy(os.path.join(args.folder, args.photo), photos_dir)
        shutil.copy(os.path.join(args.folder, bin_mask),   masks_dir)
        shutil.copy(os.path.join(args.folder, lm_txt),     landmarks_dir)

        # load + patch config
        with open(CONFIG_PATH) as f:
            cfg = yaml.safe_load(f)

        cfg['TEST_INPAINT_IMAGE_FLIST']    = [os.path.join(photos_dir,    args.photo)]
        cfg['TEST_MASK_FLIST']             = [os.path.join(masks_dir,     bin_mask)]
        cfg['TEST_INPAINT_LANDMARK_FLIST'] = [os.path.join(landmarks_dir, lm_txt)]

        # write temp config
        tmp_cfg = os.path.join(tmp,'config.yml')
        with open(tmp_cfg,'w') as f:
            yaml.safe_dump(cfg, f)

        # backup & overwrite
        backup = CONFIG_PATH + '.bak'
        shutil.copy(CONFIG_PATH, backup)
        try:
            shutil.copy(tmp_cfg, CONFIG_PATH)
            subprocess.run(['python3','test.py'], check=True)
        finally:
            shutil.copy(backup, CONFIG_PATH)
            os.remove(backup)

    print("Done.")

if __name__ == '__main__':
    main()
