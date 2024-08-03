import cv2
import glob
import os
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import json
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
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.fore = FaceMetrics()
        self.chk = FaceMetrics()

    def update_metrics(self, area, oil_all, oil_max, count, dense, zone="fore"):
        if zone == "fore":
            self.fore = FaceMetrics(area, oil_all, oil_max, count, dense)
            self.shared_state['fore'] = self.fore
        elif zone == "chk":
            self.chk = FaceMetrics(area, oil_all, oil_max, count, dense)
            self.shared_state['chk'] = self.chk
        return zone, FaceMetrics(area, oil_all, oil_max, count, dense)

class PZFace:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.PZfore = FaceMetrics()
        self.PZchk = FaceMetrics()

    def update_metrics(self, area, oil_all, oil_max, count, dense, zone="PZfore"):
        if zone == "PZfore":
            self.PZfore = FaceMetrics(area, oil_all, oil_max, count, dense)
            self.shared_state['PZfore'] = self.PZfore
        elif zone == "PZchk":
            self.PZchk = FaceMetrics(area, oil_all, oil_max, count, dense)
            self.shared_state['PZchk'] = self.PZchk
        return zone, FaceMetrics(area, oil_all, oil_max, count, dense)
    
def summary(shared_state,src):
    score = Score()
    
    oil_score = 0
    normal_score = 0
    dry_score = 0
    
    fore_oil_all = shared_state['fore'].oil_allP
    PZfore_oil_all = shared_state['PZfore'].oil_allP
    chk_oil_all = shared_state['chk'].oil_allP
    fore_dense = shared_state['fore'].dense
    Pzfore_dense = shared_state['PZfore'].dense
    fore_count = shared_state['fore'].count
    PZfore_count = shared_state['PZfore'].count
    
    score.sc1 = score.sc_fore(fore_oil_all)
    score.sc2 = score.sc_foreP(PZfore_oil_all)
    score.sc3 = score.sc_cheek(chk_oil_all)
    score.sc4 = score.sc_foreD(fore_dense)
    score.sc5 = score.sc_foreS(fore_dense)
    score.sc6 = score.sc_decreaseP(PZfore_count, fore_count)
    score.sc7 = score.sc_denseP(fore_dense, Pzfore_dense)
    
    if score.sc1 == 0:
        normal_score += 1
    elif score.sc1 == 1:
        oil_score += 1
        normal_score += 1
        dry_score += 1
    elif score.sc1 == 2:
        oil_score += 1
        dry_score += 1
        
    if score.sc2 == 0:
        normal_score += 1
    elif score.sc2 == 1:
        oil_score += 1
        normal_score += 1
        dry_score += 1
    elif score.sc2 == 2:
        oil_score += 1
        
    if score.sc3 == 0:
        normal_score += 1
    elif score.sc3 == 1:
        oil_score += 1
        normal_score += 1
        dry_score += 1
    elif score.sc3 == 2:
        oil_score += 1
        dry_score += 1
        
    if score.sc4:
        oil_score += 1
        normal_score += 0.5
        dry_score += 1
    else:
        normal_score += 1
        
    if score.sc5:
        oil_score += 1
        normal_score += 0.5
        dry_score += 1
    else:
        normal_score += 1
        dry_score += 1
        oil_score += 0.5
    
    if score.sc6:
        oil_score += 1
        normal_score += 0.5
        dry_score += 0.5
    else:
        normal_score += 1
        dry_score += 1
        oil_score += 0.5
    
    if score.sc7:
        oil_score += 0.5
        normal_score += 0.5
        dry_score += 1
    else:
        normal_score += 1
        dry_score += 0.5
        oil_score += 1
            
    def get_highest_score_class(oil_score, normal_score, dry_score):
        # Determine the highest score and corresponding skin type
        scores = {'oily_score': oil_score, 'normal_score': normal_score, 'dry_score': dry_score}
        highest_score_class = max(scores, key=scores.get)
        return highest_score_class.replace('_score', '')  # Return the skin type with the highest score

    # Example usage within the summary function
    highest_score_class = get_highest_score_class(oil_score, normal_score, dry_score)
    
    return {
        "filename": src,
        "sc_fore": score.sc1,
        "sc_foreP": score.sc2,
        "sc_cheek": score.sc3,
        "sc_foreD": score.sc4,
        "sc_foreS": score.sc5,
        "sc_decreaseP": score.sc6,
        "sc_denseP": score.sc7,
        "label": highest_score_class
    }

shared_state = {}

def apply_clahe(src, write, model):
    modes = ['normal', 'posterize']
    
    for poster in modes:            
        mp_face_mesh = mp.solutions.face_mesh

        image = cv2.imread(src)
        image = resize_with_aspect_ratio(image, width=500)
        if poster == 'posterize':
            image = posterize(image)
            
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
                         
                    
                    fore_dense = apply_model(fore_img2, model)
                    chk_dense = apply_model(chk_img2, model)
                    
                    fore_oil_all, fore_oil_max, fore_img3, fore_edge = oil_area(fore_img2)
                    chk_oil_all, chk_oil_max, chk_img3, chk_edge = oil_area(chk_img2)
                    
                    face = Face(shared_state)
                    pzface = PZFace(shared_state)
                    
                    if poster == 'posterize':
                        pzface.update_metrics(fore_area, fore_oil_all, fore_oil_max, contours_fore, fore_dense, zone="PZfore")
                        pzface.update_metrics(chk_area, chk_oil_all, chk_oil_max, contours_chk, chk_dense, zone="PZchk")
                    
                        if write:
                            summary_data = summary(shared_state, src)
                            json_data = json.dumps(summary_data, indent=4)
                            json_file_name = "Jdataset.json"
                            with open(json_file_name, "a") as f:
                                f.write(json_data + ",\n")
                                
                    else:
                        face.update_metrics(fore_area, fore_oil_all, fore_oil_max, contours_fore, fore_dense, zone="fore")
                        face.update_metrics(chk_area, chk_oil_all, chk_oil_max, contours_chk, chk_dense, zone="chk")
                

                else:
                    print("No face landmarks detected")
        except Exception as e:
            print(f"Error processing image: {e}")


model = YOLO("od_material/TFAPI_Face_Detection/XML_clahe_yolov8n.pt")
desti_f = "output"
classes = ['dry','normal','oily']
src = "svm/Train/normal/55.jpg"

def run(src, write, model):
    if write:
        for i in classes:
            src1 = 'svm/Train/'+i+'/*.*'
            for j in glob.glob(src1):
                apply_clahe(j,True,model)
    else:
        
        apply_clahe(src, False, model)
        summary(shared_state)

run(src, True, model)
