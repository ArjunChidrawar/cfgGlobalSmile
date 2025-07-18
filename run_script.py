import os
from PIL import Image
import face_alignment
from skimage import io
import cv2
import numpy as np
import argparse
import shutil
import subprocess
import tempfile
import yaml

folder = "/Users/tinazhang/Desktop/projects/cfgGlobalSmile/script_test"
photo = 'photo_01.jpg'
masked = 'masked_01.jpg'


# converting photo and mask to jpg if they aren't already
for fname in os.listdir(folder):
    if not fname.lower().endswith(".png"):
        continue

    png_path = os.path.join(folder, fname)
    jpg_name = os.path.splitext(fname)[0] + ".jpg"
    jpg_path = os.path.join(folder, jpg_name)

    # open & convert
    with Image.open(png_path) as im:
        rgb = im.convert("RGB")
        rgb.save(jpg_path, quality=95)

    # remove original
    os.remove(png_path)

# --------------------------
# Generate Landmark Photo
# --------------------------
fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                  flip_input=False, device='cpu')
file_path = os.path.join(folder, photo)
img = io.imread(file_path)
preds = fa.get_landmarks(img)
if preds is None or len(preds) == 0:
    print(f"No faces detected in {photo}.")

# For simplicity, assume only 1 face:
landmarks = preds[0]  # shape: (68, 2)

# PART A: Write the landmark points onto the image
img_landmarks = np.copy(img)
for (x, y) in landmarks:
    cv2.circle(img_landmarks, (int(x), int(y)), 2, (0, 255, 0), -1)

img_bgr = cv2.cvtColor(img_landmarks, cv2.COLOR_RGB2BGR)
landmarked_jpg_path = os.path.join(
    folder,
    photo.replace('.jpg', '_landmark.jpg')
)
cv2.imwrite(landmarked_jpg_path, img_bgr)

# PART B: Save the x,y coords in a txt file
# Flatten them into [x1, y1, x2, y2, ... , x68, y68]
flat_landmarks = landmarks.flatten()
line = " ".join(str(v) for v in flat_landmarks)

# Create the txt file path
txt_file_path = os.path.join(
    folder,
    photo.replace('.jpg', '_landmark.txt')
)
with open(txt_file_path, "w") as f:
    f.write(line + "\n")  # one line of coords

landmark_text = 'photo_01_landmark.txt'

print(f'Finished generating landmark...')



# --------------------------
# Generate Binary Mask Image
# --------------------------
masked_file_path = os.path.join(folder, masked)
img = cv2.imread(masked_file_path)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_, mask = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
cv2.imwrite(os.path.join(folder,'binary_mask.jpg'), mask)

binary_mask = 'binary_mask.jpg'

print(f'Finished generating binary mask...')

# ----------------------------------
# Make Flist Files and Set up Config
# ----------------------------------
config = '/Users/tinazhang/Desktop/projects/cfgGlobalSmile/checkpoints/config.yml'

def make_flist(tmp_dir, name, src_dir, filename):
    """Creates a one‐line flist file under tmp_dir/name.txt pointing to src_dir/filename"""
    flist_path = os.path.join(tmp_dir, name)
    with open(flist_path, 'w') as f:
        f.write(os.path.join(src_dir, filename) + '\n')
    return flist_path

with tempfile.TemporaryDirectory() as tmp:
    # 1) create subfolders
    photos_dir    = os.path.join(tmp, 'photos');    os.makedirs(photos_dir)
    masks_dir     = os.path.join(tmp, 'masks');     os.makedirs(masks_dir)
    landmarks_dir = os.path.join(tmp, 'landmarks'); os.makedirs(landmarks_dir)

    # 2) copy the three user files into their folders
    shutil.copy(os.path.join(folder, photo),    photos_dir)
    shutil.copy(os.path.join(folder, binary_mask),   masks_dir)
    shutil.copy(os.path.join(folder, landmark_text), landmarks_dir)

    # 3) generate flist files (one path per file)
    # photo_flist    = make_flist(tmp, 'cleft_input.txt',            photos_dir,    photo)
    # mask_flist     = make_flist(tmp, 'cleft_mask.txt',             masks_dir,     masked)
    # landmark_flist = make_flist(tmp, 'cleft_landmarks_flist.txt',  landmarks_dir, landmark_text)

    # 4) load & patch your YAML config
    with open(config) as f:
        cfg = yaml.safe_load(f)

    # cfg['TEST_INPAINT_IMAGE_FLIST']       = photo_flist
    # cfg['TEST_MASK_FLIST']                = mask_flist
    # cfg['TEST_INPAINT_LANDMARK_FLIST']    = landmark_flist
    cfg['TEST_INPAINT_IMAGE_FLIST']    = [ os.path.join(photos_dir,     photo) ]
    cfg['TEST_MASK_FLIST']             = [ os.path.join(masks_dir,      binary_mask) ]
    cfg['TEST_INPAINT_LANDMARK_FLIST'] = [ os.path.join(landmarks_dir,  landmark_text) ]

    # 5) write out a temp config
    tmp_cfg = os.path.join(tmp, 'config.yml')
    with open(tmp_cfg, 'w') as f:
        yaml.safe_dump(cfg, f)

    canonical_cfg = '/Users/tinazhang/Desktop/projects/cfgGlobalSmile/checkpoints/config.yml'

    # make a backup name
    backup_cfg = canonical_cfg + '.bak'

    # 1) back up the original
    shutil.copy(canonical_cfg, backup_cfg)

    try:
        shutil.copy(tmp_cfg, canonical_cfg)
        subprocess.run(['python3', 'test.py'], check=True)

    finally:
        shutil.copy(backup_cfg, canonical_cfg)
        os.remove(backup_cfg)

# --------------------------
# Set up Config File
# --------------------------

# shutil.copy(os.path.join(folder, photo),  photos_dir)
# shutil.copy(os.path.join(folder, masked),   masks_dir)
# shutil.copy(os.path.join(folder, masked), landmarks_dir)

# with open(config) as f:
#     cfg = yaml.safe_load(f)

# cfg['TEST_INPAINT_IMAGE_FLIST']  = photo_flist
# cfg['TEST_MASK_FLIST']  = mask_flist
# cfg['TEST_INPAINT_LANDMARK_FLIST']  = landmark_flist

# tmp_cfg = os.path.join(tmp, 'config.yml')
# with open(tmp_cfg, 'w') as f:
#     yaml.safe_dump(cfg, f)



# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument('--config',    required=True,
#                    help='Path to your original config.yml')
#     p.add_argument('--photo',     required=True,
#                    help='Path to the user’s photo file')
#     p.add_argument('--mask',      required=True,
#                    help='Path to the user’s mask file')
#     p.add_argument('--landmark',  default=None,
#                    help='(Optional) Path to a landmark file')
#     p.add_argument('--runner',    required=True,
#                    help='Command to run your pipeline, e.g. "python process.py"')
#     return p.parse_args()

# def main():
#     args = parse_args()

#     # 1. Create a temp “sandbox” with three subfolders
#     with tempfile.TemporaryDirectory() as tmp:
#         photos_dir   = os.path.join(tmp, 'photos')
#         masks_dir    = os.path.join(tmp, 'masks')
#         landmarks_dir= os.path.join(tmp, 'landmarks')
#         os.makedirs(photos_dir);    os.makedirs(masks_dir);    os.makedirs(landmarks_dir)

#         # 2. Copy user files in
#         shutil.copy(args.photo,  photos_dir)
#         shutil.copy(args.mask,   masks_dir)
#         if args.landmark:
#             shutil.copy(args.landmark, landmarks_dir)

#         # 3. Load + patch config.yml
#         with open(args.config) as f:
#             cfg = yaml.safe_load(f)
#         # — adjust these keys to match your config structure:
#         cfg['data']['photos_folder']    = photos_dir
#         cfg['data']['masks_folder']     = masks_dir
#         cfg['data']['landmarks_folder'] = landmarks_dir

#         # 4. Dump out a new config in the temp dir
#         tmp_cfg = os.path.join(tmp, 'config.yml')
#         with open(tmp_cfg, 'w') as f:
#             yaml.safe_dump(cfg, f)

#         # 5. Run your pipeline against the patched config
#         cmd = args.runner.split() + ['--config', tmp_cfg]
#         subprocess.run(cmd, check=True)

    # when you exit the with-block, all of tmp/* is deleted automatically

# if __name__ == '__main__':
#     main()
