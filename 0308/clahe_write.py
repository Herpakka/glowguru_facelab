import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import glob
import multiprocessing as mp_pool
import os
from helper import draw_landmarks_Tzone, posterize

def apply_clahe(src):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load the input image
    src = src.replace("\\","/")
    image = cv2.imread(src)
    # image = posterize(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try:
        # Initialize the FaceMesh model
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            if mp_face_mesh:
                # Process the image to get the face landmarks
                results = face_mesh.process(rgb_image)

                # Draw the landmarks on the image
                if results.multi_face_landmarks:
                    annotated_image ,Tzone_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks,"Tzone")
                    crop_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    clahe = cv2.createCLAHE(clipLimit=5)
                    image_bw = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
                    final_img = clahe.apply(image_bw) + 30
                    _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                    edged = cv2.Canny(final_img2, 30, 200)
                    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    osrc_list = list(src.split("/"))
                    osrc_name = list(osrc_list[2].split("."))
                    osrc = "svm/train/Toutput/" + osrc_list[1] + "/" +osrc_name[0]+ "_" + str(len(contours)) +"."+ osrc_name[1]

                    try:
                        cv2.imwrite(osrc, final_img)
                        print("write : ", osrc, " complete")
                    except Exception as e:
                        print(e)
                else:
                    print("No face landmarks detected")
            else:
                print("No face landmarks detected")
    except Exception as e:
        print(e)

def process_images_in_folder(folder):
    for src in glob.glob(folder + "/*.*"):
        apply_clahe(src)

def main():
    folder = "svm/train/"
    subfolders = [folder+"dry",folder+"normal",folder+"oily"]
    for i in subfolders:
        process_images_in_folder(i)

if __name__ == "__main__":
    main()