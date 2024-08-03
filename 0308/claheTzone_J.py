import os
import cv2
import random
import json
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import mediapipe as mp
from helper import *

# Load JSON data
with open('Jdataset.json', 'r') as f:
    data = json.load(f)

# Extract image paths and labels
image_paths = [item['filename'] for item in data]
labels = {item['filename']: item['label'] for item in data}

# Randomly select 100 images
selected_images = random.sample(image_paths, 100)

# Create a 10x10 grid with specified spacing
fig, axes = plt.subplots(10, 10, figsize=(20, 50), gridspec_kw={'wspace': 0.0, 'hspace': 0.5})

for i, ax in enumerate(axes.flat):
    img = cv2.imread(selected_images[i])
    if img is None:
        print(f"Image at {selected_images[i]} not found.")
        continue
    rgb_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    final_img4 = None

    try:
        mp_face_mesh = mp.solutions.face_mesh
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            results = face_mesh.process(rgb_image)

            if results.multi_face_landmarks:
                annotated_image, Tzone_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="Tzone")
                clahe = cv2.createCLAHE(clipLimit=5)
                final_img = use_clahe(annotated_image, clahe)
                _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                final_img4 = cv2.bitwise_and(final_img, final_img2, mask=None)
    except Exception as e:
        print(f"An error occurred: {e}")

    if final_img4 is not None:
        ax.imshow(final_img4, cmap='gray')
    else:
        ax.imshow(rgb_image)
    class_label = labels[selected_images[i]]
    ax.set_title(class_label, fontsize=8)
    ax.axis('off')

plt.tight_layout()
plt.show()
