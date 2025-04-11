import face_alignment
from skimage import io
import cv2
import numpy as np
import os

input_dir = '/Users/arjunchidrawar/Desktop/cfgGlobalSmile/INCLG/celeba_hq_256'
output_dir = '/Users/arjunchidrawar/Desktop/cfgGlobalSmile/INCLG/landmarks'
txt_output_dir = '/Users/arjunchidrawar/Desktop/cfgGlobalSmile/INCLG/landmarks_txt'

os.makedirs(output_dir, exist_ok=True)
os.makedirs(txt_output_dir, exist_ok=True)

fa = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D,
                                  flip_input=False, device='cpu')

count = 0
for file_name in os.listdir(input_dir):
    if file_name.lower().endswith('.jpg'):
        file_path = os.path.join(input_dir, file_name)
        img = io.imread(file_path)
        
        # Detect landmarks (list of face arrays)
        preds = fa.get_landmarks(img)
        if preds is None or len(preds) == 0:
            print(f"No faces detected in {file_name}. Skipping.")
            continue
        
        # For simplicity, assume only 1 face:
        landmarks = preds[0]  # shape: (68, 2)
        
        # PART A: Write the landmark points onto the image
        img_landmarks = np.copy(img)
        for (x, y) in landmarks:
            cv2.circle(img_landmarks, (int(x), int(y)), 2, (0, 255, 0), -1)
        
        img_bgr = cv2.cvtColor(img_landmarks, cv2.COLOR_RGB2BGR)
        landmarked_jpg_path = os.path.join(
            output_dir,
            file_name.replace('.jpg', '_landmarks.jpg')
        )
        cv2.imwrite(landmarked_jpg_path, img_bgr)
        
        # PART B: Save the x,y coords in a txt file
        # Flatten them into [x1, y1, x2, y2, ... , x68, y68]
        flat_landmarks = landmarks.flatten()
        line = " ".join(str(v) for v in flat_landmarks)

        # Create the txt file path
        txt_file_path = os.path.join(
            txt_output_dir,
            file_name.replace('.jpg', '.txt')
        )
        with open(txt_file_path, "w") as f:
            f.write(line + "\n")  # one line of coords
        
        count += 1
        if count % 50 == 0:
            print(f'Finished processing {count} images.')