# plot graph

import cv2
import glob
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from helper import *
import matplotlib.pyplot as plt
import os

def apply_clahe(src, mode, metrics_data=None, class_name=None):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load the input image
    image = cv2.imread(src)
    image = resize_with_aspect_ratio(image, width=500)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            
            results = face_mesh.process(rgb_image)
            if results.multi_face_landmarks:
                annotated_image, Tzone_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="Tzone")
                fore_img, fore_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="fore")
                chk_img, chk_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="chk")
                
                clahe = cv2.createCLAHE(clipLimit=5)
                final_img = use_clahe(annotated_image, clahe)
                fore_img = use_clahe(fore_img, clahe)
                chk_img = use_clahe(chk_img, clahe)
                _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                _, fore_img2 = cv2.threshold(fore_img, 30, 255, cv2.THRESH_BINARY)
                _, chk_img2 = cv2.threshold(chk_img, 30, 255, cv2.THRESH_BINARY)
                                
                contours = find_contours(final_img2)
                contours_fore = find_contours(fore_img2)
                contours_chk = find_contours(chk_img2)
                
                fore_oil_all, fore_oil_max, fore_img3, fore_edge = oil_area(fore_img2)
                chk_oil_all, chk_oil_max, chk_img3, chk_edge = oil_area(chk_img2)
                
                fore_avg = round(fore_oil_all / contours_fore if contours_fore != 0 else 0, 2)
                chk_avg = round(chk_oil_all / contours_chk if contours_chk != 0 else 0, 2)
                
                # print(f'\nfore_area = {fore_area}, fore_oil_all = {fore_oil_all}, fore_oil_max = {fore_oil_max}')
                # print(f'fore count = {contours_fore}, fore_oil_all = {round((fore_oil_all*100)/fore_area,1)}%, fore_oil_max = {round((fore_oil_max*100)/fore_area,1)}%')
                # print(f'average fore = {fore_avg}\n')
                
                # print(f'chk_area = {chk_area}, chk_oil_all = {chk_oil_all}, chk_oil_max = {chk_oil_max}')
                # print(f'chk count = {contours_chk}, chk_oil_all = {round((chk_oil_all*100)/chk_area,1)}%, chk_oil_max = {round((chk_oil_max*100)/chk_area,1)}%')
                # print(f'average chk = {chk_avg}\n')
                
                if mode == "show":
                    fore_edge = cv2.cvtColor(fore_edge, cv2.COLOR_GRAY2BGR)
                    chk_edge = cv2.cvtColor(chk_edge, cv2.COLOR_GRAY2BGR)
                    both_edge = cv2.bitwise_or(fore_edge, chk_edge, mask=None)
                    both_img3 = cv2.bitwise_and(chk_img3, fore_img3, mask=None)
                    cv2.imshow("fore and chk", cv2.hconcat([both_img3, both_edge]))
                    final_img3 = cv2.bitwise_and(final_img, final_img2, mask=None)
                    final_concat = cv2.hconcat([final_img2, final_img3])
                    cv2.imshow("final", final_concat)
                    cv2.waitKey(0)
                elif mode == "write" and metrics_data is not None and class_name is not None:
                    metrics_data[class_name].append((fore_oil_all, fore_oil_max, fore_avg, chk_oil_all, chk_oil_max, chk_avg))
            else:
                print("No face landmarks detected")
    except Exception as e:
        print(e)

desti_f = "output"
classes = ['dry', 'normal', 'oily']

def run(mode, src=None):
    metrics_data = {cls: [] for cls in classes}

    if mode == "show" and src:
        apply_clahe(src, "show")
    elif mode == "write":
        for class_name in classes:
            src1 = f'svm/Train/{class_name}/*.*'
            for img_path in glob.glob(src1):
                apply_clahe(img_path, "write", metrics_data, class_name)

        plot_metrics(metrics_data)

def plot_metrics(metrics_data):
    metrics_names = ["fore_oil_all", "fore_oil_max", "fore_avg", "chk_oil_all", "chk_oil_max", "chk_avg"]
    for metric_index, metric_name in enumerate(metrics_names):
        plt.figure(figsize=(10, 6))
        for class_name, data in metrics_data.items():
            values = [metrics[metric_index] for metrics in data]
            plt.plot(values, label=class_name)
        
        plt.title(f"{metric_name} for each class")
        plt.xlabel("Sample index")
        plt.ylabel(metric_name)
        plt.legend()
        plt.grid(True)
        plt.show()

run("show","")
