import os

# paths to your two flists
flist1 = '/Users/tinazhang/Desktop/projects/cfgGlobalSmile/cleft_mask.txt'          # contains .jpg paths
flist2 = '/Users/tinazhang/Desktop/projects/cfgGlobalSmile/cleft_landmarks_flist.txt'  # contains .txt paths

# load lines
with open(flist1) as f:
    paths1 = [line.strip() for line in f if line.strip()]
with open(flist2) as f:
    paths2 = [line.strip() for line in f if line.strip()]

# strip directories and extensions, e.g. Patient_001_2
keys1 = {os.path.splitext(os.path.basename(p))[0] for p in paths1}
keys2 = {os.path.splitext(os.path.basename(p))[0] for p in paths2}

# only keep names present in both
common = keys1 & keys2

# filter original lists by that key
filtered1 = [p for p in paths1 if os.path.splitext(os.path.basename(p))[0] in common]
filtered2 = [p for p in paths2 if os.path.splitext(os.path.basename(p))[0] in common]

# overwrite files
with open(flist1, 'w') as f:
    f.write("\n".join(filtered1) + "\n")
with open(flist2, 'w') as f:
    f.write("\n".join(filtered2) + "\n")
