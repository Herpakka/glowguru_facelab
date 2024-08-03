import cv2, glob
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
from helper import resize_with_aspect_ratio, posterize, draw_landmarks_Tzone

def naming_code(og_name,allC_id,foreC_id,chkC_id):
    sp = "_"
    return og_name+sp+allC_id+sp+foreC_id+sp+chkC_id

def apply_clahe(src):
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
                annotated_image = draw_landmarks_Tzone(rgb_image, results.multi_face_landmarks)
                
                clahe = cv2.createCLAHE(clipLimit=5) #default 5
                image_bw = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
                image_bw_re = resize_with_aspect_ratio(image_bw, width=500)
                final_img = image_bw_re.copy()
                final_img = clahe.apply(image_bw_re) + 30 # default 30
                _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                
                
                edged = cv2.Canny(final_img2, 30, 200)
                contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # if len(contours) >= 1000:
                #     final_img2 = cv2.erode(final_img2, np.ones((5, 5), np.uint8) , iterations=1) 
                #     edged = cv2.Canny(final_img2, 30, 200)
                #     contours2, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                #     cv2.imshow(str(len(contours2))+"(eroded, old:)"+str(len(contours)), final_img2)
                # else:
                #     cv2.imshow(str(len(contours)), final_img2)
                # cv2.imshow("OG", image)
                # cv2.imshow("gray",image_bw_re)
                # cv2.imshow("final 1", final_img)
                # cv2.waitKey(0)
                
                # for write images
                osrc1 = list(src.split("/"))
                osrc_f = list(osrc1[2].split("."))
                osrc2 = osrc1[0]+"/"+osrc1[1]+"/"+desti_f+"/"+osrc_f[0]+"_C"+str(len(contours))+"."+osrc_f[1]
                
                x,y = final_img.shape[:2]
                if (x <= 3*y) and (y <= 3*x):
                    if len(contours) >= 1000:
                        final_img2 = cv2.erode(final_img2, np.ones((5, 5), np.uint8) , iterations=1) 
                    final_img3 = cv2.bitwise_and(final_img,final_img2,mask=None)
                    cv2.imwrite(osrc2,final_img3)
                    print("file write in"+osrc2)

            else:
                print("No face landmarks detected")
    except Exception as e:
        print(e)

desti_f = "output"
classes = ['dry','normal','oily']

for i in classes:
    src = 'svm/Train/'+i+'/*.*'
    for j in glob.glob(src):
        apply_clahe(j) #mark oil/51.jpg
        # print(i)
