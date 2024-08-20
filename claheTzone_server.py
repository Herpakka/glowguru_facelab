# send image to server and get the result and send it to the client
import csv, cv2, glob
import mediapipe as mp
import numpy as np

from skimage.feature import graycomatrix, graycoprops
from skimage.color import rgb2gray
from PIL import Image

from helper import *
from ultralytics import YOLO


class FaceMetrics:
    def __init__(self, area=None, oil_all=None, oil_max=None, count=None, dense=None, spread=None):
        self.area = area
        self.oil_all = oil_all
        self.oil_max = oil_max
        self.count = count
        self.dense = dense
        self.spread = spread
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

    def update_metrics(self, area, oil_all, oil_max, count, dense, spread, zone="fore"):
        metrics = FaceMetrics(area, oil_all, oil_max, count, dense, spread)
        if zone == "fore":
            self.fore = metrics
            self.shared_state['fore'] = {
                'area': area,
                'oil_all': oil_all,
                'oil_max': oil_max,
                'count': count,
                'dense': dense,
                'spread': spread,
                'avg': metrics.avg,
                'oil_maxP': metrics.oil_maxP,
                'oil_allP': metrics.oil_allP
            }
        elif zone == "chk":
            self.chk = metrics
            self.shared_state['chk'] = {
                'area': area,
                'oil_all': oil_all,
                'oil_max': oil_max,
                'count': count,
                'dense': dense,
                'spread': spread,
                'avg': metrics.avg,
                'oil_maxP': metrics.oil_maxP,
                'oil_allP': metrics.oil_allP
            }
        return zone, metrics


class PZFace:
    def __init__(self, shared_state):
        self.shared_state = shared_state
        self.PZfore = FaceMetrics()
        self.PZchk = FaceMetrics()

    def update_metrics(self, area, oil_all, oil_max, count, dense, spread, zone="PZfore"):
        metrics = FaceMetrics(area, oil_all, oil_max, count, dense, spread)
        if zone == "PZfore":
            self.PZfore = metrics
            self.shared_state['PZfore'] = {
                'area': area,
                'oil_all': oil_all,
                'oil_max': oil_max,
                'count': count,
                'dense': dense,
                'spread': spread,
                'avg': metrics.avg,
                'oil_maxP': metrics.oil_maxP,
                'oil_allP': metrics.oil_allP
            }
        elif zone == "PZchk":
            self.PZchk = metrics
            self.shared_state['PZchk'] = {
                'area': area,
                'oil_all': oil_all,
                'oil_max': oil_max,
                'count': count,
                'dense': dense,
                'spread': spread,
                'avg': metrics.avg,
                'oil_maxP': metrics.oil_maxP,
                'oil_allP': metrics.oil_allP
            }
        return zone, metrics

class GLCM:
    def __init__(self, shared_state):
        self.contrast = 0
        self.dissimilarity = 0
        self.homogeneity = 0
        self.energy = 0
        self.correlation = 0
        self.shared_state = shared_state

    def update_metrics(self, contrast, dissimilarity, homogeneity, energy, correlation):
        self.contrast = contrast
        self.dissimilarity = dissimilarity
        self.homogeneity = homogeneity
        self.energy = energy
        self.correlation = correlation
        self.shared_state['GLCM'] = {
            'contrast': contrast,
            'dissimilarity': dissimilarity,
            'homogeneity': homogeneity,
            'energy': energy,
            'correlation': correlation
        }
        return contrast, dissimilarity, homogeneity, energy, correlation

shared_state = {}

def convertCSV(shared_state, filename="LGBM1.csv", src=""):
    # Initialize an empty dictionary to store combined metrics
    combined_row = {}
    src0 = src
    
    # Iterate through the shared state dictionary
    combined_row['filename'] = src0
    for classname, metrics in shared_state.items():
        for key, value in metrics.items():
            # Flatten the keys by prefixing the classname
            prefixed_key = f"{classname}_{key}"
            combined_row[prefixed_key] = value

    # Determine the fieldnames (keys) from the combined_row
    fieldnames = combined_row.keys()

    # Write the data to a CSV file
    with open(filename, 'a', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, fieldnames=fieldnames)

        # Write the header only if the file is empty
        if output_file.tell() == 0:
            dict_writer.writeheader()

        # Write the combined row
        dict_writer.writerow(combined_row)

def GLCM_calculation(image):
    im_frame = Image.open(image)
    image = (255 * rgb2gray(np.array(im_frame))).astype(np.uint8)
    max_gray_value = image.max()
    distances = [50]  # Offset
    angles = [np.pi/2]  # Vertical Direction
    glcm = graycomatrix(image, distances=distances, angles=angles, levels=max_gray_value + 1)
    
    contrast = graycoprops(glcm, 'contrast').flatten().astype(int)[0]
    dissimilarity = graycoprops(glcm, 'dissimilarity').flatten().astype(int)[0]
    homogeneity = graycoprops(glcm, 'homogeneity').flatten().astype(int)[0]
    energy = graycoprops(glcm, 'energy').flatten().astype(int)[0]
    correlation = graycoprops(glcm, 'correlation').flatten().astype(int)[0]
    return contrast, dissimilarity, homogeneity, energy, correlation

def apply_clahe(src, model):
    modes = ['normal', 'posterize']
    face_detected = False
    
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
                    annotated_image, tzone_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="Tzone")
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

                    fore_dense, fore_spread = apply_model(fore_img2, model)
                    chk_dense, chk_spread= apply_model(chk_img2, model)
                    
                    fore_oil_all, fore_oil_max, fore_img3, fore_edge = oil_area(fore_img2)
                    chk_oil_all, chk_oil_max, chk_img3, chk_edge = oil_area(chk_img2)
                    
                    contrast, dissimilarity, homogeneity, energy, correlation = GLCM_calculation(src)
                    glcm = GLCM(shared_state)
                    glcm.update_metrics(contrast, dissimilarity, homogeneity, energy, correlation)

                    face = Face(shared_state)
                    pzface = PZFace(shared_state)
                    
                    if poster == 'posterize':
                        pzface.update_metrics(fore_area, fore_oil_all, fore_oil_max, contours_fore, fore_dense, fore_spread, zone="PZfore")
                        pzface.update_metrics(chk_area, chk_oil_all, chk_oil_max, contours_chk, chk_dense, chk_spread, zone="PZchk")                            
                                
                    else:
                        face.update_metrics(fore_area, fore_oil_all, fore_oil_max, contours_fore, fore_dense, fore_spread, zone="fore")
                        face.update_metrics(chk_area, chk_oil_all, chk_oil_max, contours_chk, chk_dense, chk_spread, zone="chk")
                
                    face_detected = True
                else:
                    print("No face landmarks detected")
                    face_detected = False
        except Exception as e:
            print(f"Error processing image: {e}")
    if face_detected:
        convertCSV(shared_state, "LGBM1.csv",src)


model = YOLO("od_material/TFAPI_Face_Detection/XML_clahe_yolov8n.pt")
desti_f = "svm/Train"
# classes = ['normal','dry','oily']
classes = ['Female_Faces','Male_Faces']
src = "train/oily/0 (1109).jpg"


def run(src, model):
        apply_clahe(src, model)

run(src, model)