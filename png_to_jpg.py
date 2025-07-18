import os
from PIL import Image

folder = "/Users/tinazhang/Desktop/projects/cfgGlobalSmile/After_no_mask_resized"

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

    print(f"Converted: {fname} → {jpg_name}")
