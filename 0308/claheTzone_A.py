# this is experiment for finding biggest clahe area it still work on forehead

import cv2, glob
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from helper import *


def apply_clahe(src,mode):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load the input image
    image = cv2.imread(src)
    image = resize_with_aspect_ratio(image, width=500)
    # image = posterize(image)
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
                annotated_image,Tzone_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks,mode="Tzone")
                fore_img,fore_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks,mode="fore")
                chk_img,chk_area = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks,mode="chk")
                
                clahe = cv2.createCLAHE(clipLimit=5) #default 5
                final_img = use_clahe(annotated_image,clahe)
                fore_img = use_clahe(fore_img,clahe)
                chk_img = use_clahe(chk_img,clahe)
                _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                _, fore_img2 = cv2.threshold(fore_img, 30, 255, cv2.THRESH_BINARY)
                _, chk_img = cv2.threshold(chk_img, 30, 255, cv2.THRESH_BINARY)
                                
                contours = find_contours(final_img2)
                contours_fore = find_contours(fore_img2)
                contours_chk = find_contours(chk_img)
                # max_fore = fore_area(fore_img2,results.multi_face_landmarks) # def fore_img ,,fine rgb_image
                fore_oil_all, fore_oil_max, fore_img3 = oil_area(fore_img2)
                print(f'\nfore_area = {fore_area}, fore_oil_all = {fore_oil_all}, fore_oil_max = {fore_oil_max}')
                print(f'fore_area = 100%, fore_oil_all = {round((fore_oil_all*100)/fore_area,1)}%, fore_oil_max = {round((fore_oil_max*100)/fore_area,1)}%\n')
                
                if mode == "show":
                    if contours >= 1000:
                        final_img2 = cv2.erode(final_img2, np.ones((5, 5), np.uint8) , iterations=1) 
                        edged = cv2.Canny(final_img2, 30, 200)
                        contours2, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                        cv2.imshow(str(contours2)+"(eroded, old:)"+str(contours), final_img2)
                    else:
                        cv2.imshow(f'final img 2', final_img2)
                    final_img3 = cv2.bitwise_and(final_img,final_img2,mask=None)
                    cv2.imshow("final 1", final_img3) #def final_img3
                    cv2.imshow(f'fore {fore_area}, {contours_fore}',fore_img)
                    cv2.imshow(f'fore clahe',fore_img3) # replace with oil area
                    cv2.waitKey(0)
                elif mode == "write":
                    # for write images
                    osrc1 = list(src.split("/"))
                    osrc_f = list(osrc1[2].split("."))
                    code = naming_code(osrc_f[0],str(contours),str(contours_fore),str(contours_chk))
                    osrc2 = osrc1[0]+"/"+osrc1[1]+"/"+desti_f+"/"+code+"."+osrc_f[1]
                    x,y = final_img.shape[:2]
                    if (x <= 3*y) and (y <= 3*x):
                        if contours >= 1000:
                            final_img2 = cv2.erode(final_img2, np.ones((5, 5), np.uint8) , iterations=1) 
                        final_img3 = cv2.bitwise_and(final_img,final_img2,mask=None)
                        cv2.imwrite(osrc2,final_img3)
                        print("file write in"+osrc2)
                        # print(osrc2+" ",contours_fore,contours_chk)

            else:
                print("No face landmarks detected")
    except Exception as e:
        print(e)
        
desti_f = "output"
classes = ['dry','normal','oily']

def run(mode,src):
    if mode == "show":
        apply_clahe(src,"show")
    elif mode == "write":
        for i in classes:
            src1 = 'svm/Train/'+i+'/*.*'
            for j in glob.glob(src1,"write"):
                apply_clahe(j) #mark oil/51.jpg
                # print(i)

run("show","svm/Train/oily/11.jpg")