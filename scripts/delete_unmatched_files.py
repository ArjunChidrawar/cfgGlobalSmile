import os

# Paths to the file lists
mask_flist = 'augmented_mask_flist.txt'
no_mask_flist = 'augmented_no_mask_flist.txt'
landmarks_flist = 'augmented_landmarks_flist.txt'

# Helper to read file list into a set (strip whitespace)
def read_flist(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]

def get_basename_no_ext(filename):
    base = os.path.basename(filename)
    name = os.path.splitext(base)[0]
    return name

# Read all file lists
mask_files = read_flist(mask_flist)
no_mask_files = read_flist(no_mask_flist)
landmark_files = read_flist(landmarks_flist)

# Build set of basenames (no extension) from landmarks
landmark_basenames = set(get_basename_no_ext(f) for f in landmark_files)

def filter_flist(flist, flist_path):
    kept = []
    removed = []
    for file_path in flist:
        basename = get_basename_no_ext(file_path)
        if basename in landmark_basenames:
            kept.append(file_path)
        else:
            removed.append(file_path)
    # Overwrite the flist file with only the kept entries
    with open(flist_path, 'w') as f:
        for line in kept:
            f.write(line + '\n')
    for line in removed:
        print(f"Removed from {flist_path}: {line}")

filter_flist(mask_files, mask_flist)
filter_flist(no_mask_files, no_mask_flist)
