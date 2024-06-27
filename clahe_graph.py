import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt
import glob

def sort_points_clockwise(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    # Sort points based on the angle with respect to the centroid
    sorted_points = sorted(points, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
    return np.array(sorted_points, dtype=np.int32)

def draw_landmarks_on_image(rgb_image, face_landmarks_list):
    annotated_image = np.copy(rgb_image)
    face_rect = []

    # Loop through the detected faces to visualize.
    for face_landmarks in face_landmarks_list:
        # Draw the face landmarks.
        mask = np.zeros_like(rgb_image)
        face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        face_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks.landmark
        ])
        
        # face polygon
        face_indices = [i for t in mp.solutions.face_mesh.FACEMESH_FACE_OVAL for i in t]
        face_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in face_indices], dtype=np.int32)
        face_polygon = sort_points_clockwise(face_polygon) # เรียง polygon ใบหน้าตามเข็มนาฬิกา
        
        # eyes polygon
        L_eye_indices = [i for t in mp.solutions.face_mesh.FACEMESH_LEFT_EYE for i in t]
        L_eye_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in L_eye_indices], dtype=np.int32)
        L_eye_polygon = sort_points_clockwise(L_eye_polygon) # เรียง polygon ตาซ้ายตามเข็มนาฬิกา
        R_eye_indices = [i for t in mp.solutions.face_mesh.FACEMESH_RIGHT_EYE for i in t]
        R_eye_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in R_eye_indices], dtype=np.int32)
        R_eye_polygon = sort_points_clockwise(R_eye_polygon) # เรียง polygon ตาขวาตามเข็มนาฬิกา
        
        # mouth polygon
        mouth_indices = [i for t in mp.solutions.face_mesh.FACEMESH_LIPS for i in t]
        mouth_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in mouth_indices], dtype=np.int32)
        mouth_polygon = sort_points_clockwise(mouth_polygon) # เรียง polygon ปากตามเข็มนาฬิกา
        
        # Find bounding box of face polygon
        x, y, w, h = cv2.boundingRect(face_polygon)
        face_rect.append((x, y, w, h))  # Store bounding box coordinates
        
        # Fill the face polygon on the mask
        cv2.fillPoly(mask, [face_polygon], (255, 255, 255))
        cv2.fillPoly(mask, [L_eye_polygon], (0, 0, 0))
        cv2.fillPoly(mask, [R_eye_polygon], (0, 0, 0))
        cv2.fillPoly(mask, [mouth_polygon], (0, 0, 0))
        # Black out the area outside the face
        annotated_image = cv2.bitwise_and(annotated_image, mask)
        cv2.fillPoly(mask, [L_eye_polygon], (255, 255, 255))
        
        annotated_image = annotated_image[y:y + h, x:x + w]

    return annotated_image

def apply_clahe(src,folder):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load the input image
    image = cv2.imread(src)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    try :
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
                annotated_image = draw_landmarks_on_image(rgb_image, results.multi_face_landmarks)
                
                clahe = cv2.createCLAHE(clipLimit=5)
                image_bw = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
                final_img = image_bw.copy()
                final_img = clahe.apply(image_bw) + 30
                _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                
                edged = cv2.Canny(final_img2, 30, 200)
                contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contours = len(contours)
                if contours > 1000 :
                    contours = 1000
                if folder == folders[0]:
                    contour_oily.append(contours)
                elif folder == folders[1]:
                    contour_normal.append(contours)
                elif folder == folders[2]:
                    contour_dry.append(contours)
                
                # cv2.imshow("OG", image) 
                # cv2.imshow("grayscale", final_img) # ภาพขาวดำ
                # cv2.imshow(str(len(contours)), final_img2) # ภาพclahe
                # cv2.waitKey(0)
                
            else:
                print("No face landmarks detected")
    except Exception as e:
        print(e)

def average(lst):
    a1 =  sum(lst)/len(lst)
    a2 = round(a1,0)
    return a2

folders = ["oily","normal","dry"]
contour_oily = []
contour_normal = []
contour_dry = []
contour_sum = [contour_oily,contour_normal,contour_dry]
for x in folders:
    path = "svm\\train\\"+x+"\\*.*"
    for src in glob.glob(path):
        apply_clahe(src,x)

# plot
ox = np.array([1,len(contour_oily)])
oy = np.array(contour_oily)
nx = np.array([1,len(contour_normal)])
ny = np.array(contour_normal)
dx = np.array([1,len(contour_dry)])
dy = np.array(contour_dry)
Ko = average(contour_sum[0])
Kn = average(contour_sum[1])
Kd = average(contour_sum[2])

plt.subplot(311)
plt.plot(oy)
plt.title(f"oily average = {Ko}")
plt.subplot(312)
plt.plot(ny)
plt.title(f"normal average = {Kn}")
plt.subplot(313)
plt.plot(dy)
plt.title(f"dry average = {Kd}")
plt.tight_layout()
plt.show()
