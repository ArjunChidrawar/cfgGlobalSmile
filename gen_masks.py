import cv2
import os
import numpy as np

input_dir = "/Users/tinazhang/Desktop/projects/cfgGlobalSmile/celeba_hq_256"
output_dir = "/Users/tinazhang/Desktop/projects/cfgGlobalSmile/masked_images"

count = 0
for file_name in os.listdir(input_dir):

    if os.path.exists(os.path.join(output_dir, file_name)):
        continue
    if file_name in ["01439.jpg", "08722.jpg", "13427.jpg", ".DS_Store"]:
        print('hi')
        continue

    if count%50 == 0:
        print(f'finished processing {count} images')

    file_path = os.path.join(input_dir, file_name)

    # Load your image
    image = cv2.imread(file_path)

    # Coordinates of the rectangle:
    # (x1, y1) is the top‑left corner, (x2, y2) is the bottom‑right corner
    x1, y1 = 90,  165
    x2, y2 = 150, 205

    # Color of the mask (B, G, R). For white: (255, 255, 255)
    mask_color = (255, 255, 255)

    # Thickness = -1 means "filled rectangle"
    cv2.rectangle(image, (x1, y1), (x2, y2), mask_color, thickness=-1)



    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, file_name)

    if image is None:
        raise ValueError(f"Failed to load image—cv2.imread returned None for path: {file_path}")
    if not hasattr(image, "size") or image.size == 0:
        raise ValueError("Image array is empty (shape: {})".format(getattr(image, "shape", None)))
    cv2.imwrite(output_file, image)
    count += 1

for file_name in os.listdir(input_dir):
    file_path = os.path.join(output_dir, file_name)
    if os.path.exists(os.path.join(output_dir, file_name)):
        continue
    remove_path = os.path.join(input_dir, file_name)
    os.remove(remove_path)

for file_name in os.listdir(output_dir):
    old_path = os.path.join(output_dir, file_name)
    new_name = "masked_" + file_name
    new_path = os.path.join(output_dir, new_name)
    os.rename(old_path, new_path)

print(len(os.listdir(output_dir)))
print(len(os.listdir(input_dir)))
