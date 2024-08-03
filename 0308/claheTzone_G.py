import cv2
import glob
import os
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from helper import *
from ultralytics import YOLO
from score import Score

class FaceMetrics:
    def __init__(self, area=None, oil_all=None, oil_max=None, count=None, dense=None):
        self.area = area
        self.oil_all = oil_all
        self.oil_max = oil_max
        self.count = count
        self.dense = dense
        self.avg = oil_all / count if count and count > 0 else 0
        self.oil_maxP = self.percentage(oil_max)
        self.oil_allP = self.percentage(oil_all)
        
    def percentage(self, value):
        return (value / self.area) * 100 if self.area and self.area > 0 else 0

class Face:
    def __init__(self, src):
        self.src = src
        self.fore = FaceMetrics()
        self.chk = FaceMetrics()
        self.PZfore = FaceMetrics()
        self.PZchk = FaceMetrics()

    def update_metrics(self, area, oil_all, oil_max, count, dense, zone="fore"):
        if zone == "fore":
            self.fore = FaceMetrics(area, oil_all, oil_max, count, dense)
            print(f'fore: {self.fore.__dict__}')
        elif zone == "PZfore":
            self.PZfore = FaceMetrics(area, oil_all, oil_max, count, dense)
            print(f'PZfore: {self.PZfore.__dict__}')
        elif zone == "chk":
            self.chk = FaceMetrics(area, oil_all, oil_max, count, dense)
            print(f'chk: {self.chk.__dict__}')
        elif zone == "PZchk":
            self.PZchk = FaceMetrics(area, oil_all, oil_max, count, dense)
            print(f'PZchk: {self.PZchk.__dict__}')
        return zone, FaceMetrics(area, oil_all, oil_max, count, dense)
        
        
    def summary(self):
        score = Score()
        fore_oil_all = self.fore.oil_allP
        PZfore_oil_all = self.PZfore.oil_allP
        chk_oil_all = self.chk.oil_allP
        fore_dense = self.fore.dense
        Pzfore_dense = self.PZfore.dense
        fore_count = self.fore.count
        PZfore_count = self.PZfore.count
        print(f'\nfore_oil_all :{fore_oil_all} {type(fore_oil_all)}\nPZfore_oil_all :{PZfore_oil_all} {type(PZfore_oil_all)}\nchk_oil_all :{chk_oil_all} {type(chk_oil_all)}\nfore_dense :{fore_dense} {type(fore_dense)}\nPzfore_dense :{Pzfore_dense} {type(Pzfore_dense)}\nfore_count :{fore_count} {type(fore_count)}\nPZfore_count :{PZfore_count} {type(PZfore_count)}')
        
        print(self.fore.__dict__)
        
        
        score.sc1 = score.sc_fore(fore_oil_all)
        score.sc2 = score.sc_foreP(PZfore_oil_all)
        score.sc3 = score.sc_cheek(chk_oil_all)
        score.sc4 = score.sc_foreD(fore_dense)
        score.sc5 = score.sc_foreS(fore_dense)
        score.sc6 = score.sc_decreaseP(PZfore_count, fore_count)
        score.sc7 = score.sc_denseP(fore_dense, Pzfore_dense)
        print(f'\nScore:\nsc1: {score.sc1}\nsc2: {score.sc2}\nsc3: {score.sc3}\nsc4: {score.sc4}\nsc5: {score.sc5}\nsc6: {score.sc6}\nsc7: {score.sc7}')
        return score

def apply_clahe(src, mode, model, poster=False):
    if poster == True:
        print("-------------------------------------------------------------")
        print("Posterize\n")
    else:
        print("-------------------------------------------------------------")
        print("Normal\n")
    mp_face_mesh = mp.solutions.face_mesh

    image = cv2.imread(src)
    image = resize_with_aspect_ratio(image, width=500)
    if poster == True:
        image = posterize(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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
                     
                if mode == "show":
                    fore_dense = apply_model(fore_img2,model)
                    chk_dense = apply_model(chk_img2,model)
                    
                    fore_oil_all, fore_oil_max, fore_img3, fore_edge = oil_area(fore_img2)
                    chk_oil_all, chk_oil_max, chk_img3, chk_edge = oil_area(chk_img2)
                    
                    face = Face(src)
                    
                    if poster:
                        face.update_metrics(fore_area, fore_oil_all, fore_oil_max, contours_fore, fore_dense, zone="PZfore")
                        face.update_metrics(chk_area, chk_oil_all, chk_oil_max, contours_chk, chk_dense, zone="PZchk")
                    else:
                        face.update_metrics(fore_area, fore_oil_all, fore_oil_max, contours_fore, fore_dense, zone="fore")
                        face.update_metrics(chk_area, chk_oil_all, chk_oil_max, contours_chk, chk_dense, zone="chk")
                    
                
                elif mode == "write":
                    # osrc1 = list(src.split("/"))
                    # osrc_f = list(osrc1[2].split("."))
                    # code = naming_code(osrc_f[0], str(contours), str(contours_fore), str(contours_chk))
                    # osrc2 = osrc1[0] + "/" + osrc1[1] + "/" + desti_f + "/" + code + "." + osrc_f[1]
                    # x, y = final_img.shape[:2]
                    # if (x <= 3 * y) and (y <= 3 * x):
                    #     if contours >= 1000:
                    #         final_img2 = cv2.erode(final_img2, np.ones((5, 5), np.uint8), iterations=1)
                    #     final_img3 = cv2.bitwise_and(final_img, final_img2, mask=None)
                    #     cv2.imwrite(osrc2, final_img2)
                    #     print("file write in" + osrc2)
                    print("Write mode not implemented yet")

            else:
                print("No face landmarks detected")

src = "svm/Train/oily/4.jpg"
model = YOLO("od_material/TFAPI_Face_Detection/XML_clahe_yolov8n.pt")
apply_clahe(src, "show", model, poster=False)
apply_clahe(src, "show", model, poster=True)
