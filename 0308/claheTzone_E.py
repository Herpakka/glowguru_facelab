import cv2
import glob
import os
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from helper import *
from ultralytics import YOLO
from score import *


class Face:
    # มีจุดบนหน้าผาก sc_fore
    # ใช้posteriseแล้วก็ยังมีจุด sc_foreP
    # มีจุดบนแก้ม sc_cheek
    # จุดเกาะกลุ่มบนหน้าผาก sc_foreD
    # จุดกระจายบนหน้าผาก sc_foreS
    # ใช้posteriseแล้วจุดลด sc_decreaseP
    # ใช้posteriseแล้วจุดเกาะกลุ่ม sc_denseP
    
    # 1:0 2:1.5 3:3 4:6
    
    # 19.5  มัน
    # 10.5  ปกติ
    # 18    แห้ง
    sc1 = 0
    sc2 = 0
    sc3 = 0
    sc4 = False
    sc5 = False
    sc6 = 0
    sc7 = 0
    
    def __init__(self, src, 
                 fore_area, fore_oil_all, fore_oil_max, fore_oil_allP, fore_oil_maxP, fore_count, fore_avg, fore_dense, 
                 chk_area, chk_oil_all, chk_oil_max, chk_oil_allP, chk_oil_maxP, chk_count, chk_avg, chk_dense):
        self.src = src
        self.fore_area = fore_area
        self.fore_oil_all = fore_oil_all
        self.fore_oil_allP = fore_oil_allP
        self.fore_count = fore_count
        self.fore_oil_max = fore_oil_max
        self.fore_oil_maxP = fore_oil_maxP
        self.fore_avg = fore_avg
        self.fore_dense = fore_dense
        
        self.chk_area = chk_area
        self.chk_oil_all = chk_oil_all
        self.chk_oil_allP = chk_oil_allP
        self.chk_count = chk_count
        self.chk_oil_max = chk_oil_max
        self.chk_oil_maxP = chk_oil_maxP
        self.chk_avg = chk_avg
        self.chk_dense = chk_dense

    def calculate_forehead_density(self):
        if self.fore_area > 0:
            self.fore_dense = self.fore_oil_all / self.fore_area
        else:
            self.fore_dense = 0

    def calculate_cheek_density(self):
        if self.chk_area > 0:
            self.chk_dense = self.chk_oil_all / self.chk_area
        else:
            self.chk_dense = 0

    def update_forehead_metrics(self, area, oil_all, oil_max, count, dense):
        # print(f'\nclass Face.forehead:\nfore_area:{self.fore_area}\nfore_oil_all:{self.fore_oil_all} ~{self.fore_oil_allP}%\nfore_oil_max:{self.fore_oil_max} ~{self.fore_oil_maxP}%\nfore_count:{self.fore_count}\nfore_dense:{self.fore_dense}')
        self.fore_area = area
        self.fore_oil_all = oil_all
        self.fore_oil_max = oil_max
        self.fore_count = count
        self.fore_dense = dense
        self.fore_avg = oil_all / count if count > 0 else 0
        self.fore_oil_maxP = percentage(area,oil_max)
        self.fore_oil_allP = percentage(area,oil_all)
        print(f'\nupdated class Face.forehead:\nfore_area:{area}\nfore_oil_all:{oil_all} ~{self.fore_oil_allP}%\nfore_oil_max:{oil_max} ~{self.fore_oil_maxP}%\nfore_count:{count}\nfore_dense:{dense}')
        # self.calculate_forehead_density()

    def update_cheek_metrics(self, area, oil_all, oil_max, count, dense):
        # print(f'\nclass Face.cheeks:\nchk_area:{self.chk_area}\nchk_oil_all:{self.chk_oil_all} ~{self.chk_oil_allP}%\nfore_oil_max:{self.chk_oil_max} ~{self.chk_oil_maxP}%\nchk_count:{self.chk_count}\nchk_dense:{self.chk_dense}')
        self.chk_area = area
        self.chk_oil_all = oil_all
        self.chk_oil_max = oil_max
        self.chk_count = count
        self.chk_dense = dense
        self.chk_avg = oil_all / count if count > 0 else 0
        self.chk_oil_maxP = percentage(area,oil_max)
        self.chk_oil_allP = percentage(area,oil_all)
        print(f'\nupdated class Face.cheeks:\nchk_area:{area}\nchk_oil_all:{oil_all} ~{self.chk_oil_allP}%\nfore_oil_max:{oil_max} ~{self.chk_oil_maxP}%\nchk_count:{count}\nchk_dense:{dense}')
        # self.calculate_cheek_density()
        self.score()
    
    
    def sc_fore(self, fore_oil_allP):
        self.sc1 = 0 if self.fore_oil_allP <= 10 else 1 if self.fore_oil_allP <= 50 else 2
        return self.sc1

    def sc_cheek(self, chk_oil_allP):
        self.sc3 = 0 if self.chk_oil_allP <= 10 else 1 if self.chk_oil_allP <= 50 else 2
        return self.sc3

    def sc_foreD(self, fore_dense):
        self.sc4 = self.fore_dense[0] > 0
        return self.sc4

    def sc_foreS(self, fore_dense):
        self.sc5 = self.fore_dense[1] > 0
        return self.sc5
    
    def score(self):
        self.sc_fore(self)
        self.sc_cheek(self)
        self.sc_foreD(self)
        self.sc_foreS(self)
        print(f'\nมีจุดบนหน้าผาก {self.sc1}')
        print(f'มีจุดบนแก้ม {self.sc3}')
        print(f'จุดเกาะกลุ่มบนหน้าผาก {self.sc4}')
        print(f'จุดกระจายบนหน้าผาก {self.sc5}')
    

def apply_clahe(src, mode, model, poster=False):
    if poster == True:
        print("-------------------------------------------------------------")
        print("Posterize")
    else:
        print("-------------------------------------------------------------")
        print("Normal")
    mp_face_mesh = mp.solutions.face_mesh

    # Load the input image
    image = cv2.imread(src)
    image = resize_with_aspect_ratio(image, width=500)
    if poster == True:
        image = posterize(image)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try:
        # Initialize the FaceMesh model
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:

            # Process the image to get the face landmarks
            results = face_mesh.process(rgb_image)

            # Draw the landmarks on the image
            if results.multi_face_landmarks:
                annotated_image, Tzone_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="Tzone")
                fore_img, fore_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="fore")
                chk_img, chk_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks, mode="chk")

                clahe = cv2.createCLAHE(clipLimit=5)  # default 5
                final_img = use_clahe(annotated_image, clahe)
                fore_img = use_clahe(fore_img, clahe)
                chk_img = use_clahe(chk_img, clahe)
                _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                _, fore_img2 = cv2.threshold(fore_img, 30, 255, cv2.THRESH_BINARY)
                _, chk_img2 = cv2.threshold(chk_img, 30, 255, cv2.THRESH_BINARY)

                contours = find_contours(final_img2)
                contours_fore = find_contours(fore_img2)
                contours_chk = find_contours(chk_img2)
                
                face = Face(src, fore_area, 0, 0, 0, 0, 0, 0, 0, chk_area, 0, 0, 0, 0, 0, 0, 0,)

                if mode == "show":
                    print("forehead dense:")
                    fore_dense = apply_model(fore_img2,model)
                    chk_dense = apply_model(chk_img2,model)
                    
                    fore_oil_all, fore_oil_max, fore_img3, fore_edge = oil_area(fore_img2)
                    chk_oil_all, chk_oil_max, chk_img3, chk_edge = oil_area(chk_img2)

                    fore_avg = round(fore_oil_all / contours_fore if contours_fore != 0 else 0, 2)
                    chk_avg = round(chk_oil_all / contours_chk if contours_chk != 0 else 0, 2)

                    face.update_forehead_metrics(fore_area, fore_oil_all,fore_oil_max, contours_fore, fore_dense)
                    face.update_cheek_metrics(chk_area, chk_oil_all, chk_oil_max, contours_chk, chk_dense)

                    # print(f'fore_area = {face.fore_area}, fore_oil_all = {face.fore_oil_all}, fore_oil_max = {fore_oil_max}')
                    # print(f'fore count = {face.fore_count}, fore_oil_all = {round((face.fore_oil_all * 100) / face.fore_area, 1)}%, fore_oil_max = {round((fore_oil_max * 100) / face.fore_area, 1)}%')
                    # print(f'average fore = {face.fore_avg}\n')

                    # print(f'chk_area = {face.chk_area}, chk_oil_all = {face.chk_oil_all}, chk_oil_max = {chk_oil_max}')
                    # print(f'chk count = {face.chk_count}, chk_oil_all = {round((face.chk_oil_all * 100) / face.chk_area, 1)}%, chk_oil_max = {round((chk_oil_max * 100) / face.chk_area, 1)}%')
                    # print(f'average chk = {face.chk_avg}\n')

                    # fore_edge = cv2.cvtColor(fore_edge, cv2.COLOR_GRAY2BGR)
                    # chk_edge = cv2.cvtColor(chk_edge, cv2.COLOR_GRAY2BGR)
                    # both_edge = cv2.bitwise_or(fore_edge, chk_edge, mask=None)
                    # both_img3 = cv2.bitwise_and(chk_img3, fore_img3, mask=None)
                    # final_img3 = cv2.bitwise_and(final_img, final_img2, mask=None)
                    # final_concat = cv2.hconcat([both_img3, both_edge, cv2.cvtColor(final_img3, cv2.COLOR_GRAY2BGR)])
                    # cv2.imshow("final", final_concat)  # def final_img3
                    # cv2.waitKey(0)

                elif mode == "write":
                    osrc1 = list(src.split("/"))
                    osrc_f = list(osrc1[2].split("."))
                    code = naming_code(osrc_f[0], str(contours), str(contours_fore), str(contours_chk))
                    osrc2 = osrc1[0] + "/" + osrc1[1] + "/" + desti_f + "/" + code + "." + osrc_f[1]
                    x, y = final_img.shape[:2]
                    if (x <= 3 * y) and (y <= 3 * x):
                        if contours >= 1000:
                            final_img2 = cv2.erode(final_img2, np.ones((5, 5), np.uint8), iterations=1)
                        final_img3 = cv2.bitwise_and(final_img, final_img2, mask=None)
                        cv2.imwrite(osrc2, final_img2)
                        print("file write in" + osrc2)

            else:
                print("No face landmarks detected")
    except Exception as e:
        print(e)

desti_f = "Toutput"
classes = ['dry', 'normal', 'oily']
model = YOLO("od_material/TFAPI_Face_Detection/XML_clahe_yolov8n.pt")

def run(mode, src):
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    if mode == "show":
        apply_clahe(src, "show", model, False)
    elif mode == "write":
        for i in classes:
            src1 = 'svm/Train/' + i + '/*.*'
            for j in glob.glob(src1):
                apply_clahe(j, "write", model)


run("show", "svm/Train/dry/16.jpg")
