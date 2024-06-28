import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import matplotlib.pyplot as plt

def resize_with_aspect_ratio(image, width=None, height=None):
    # Get the original image dimensions
    h, w = image.shape[:2]

    # Calculate the aspect ratio
    aspect_ratio = w / h

    if width is None:
        # Calculate height based on the specified width
        new_height = int(height / aspect_ratio)
        resized_image = cv2.resize(image, (height, new_height))
    else:
        # Calculate width based on the specified height
        new_width = int(width * aspect_ratio)
        resized_image = cv2.resize(image, (new_width, width))

    return resized_image

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
        lip_indice = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91,146]
        mouth_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in lip_indice], dtype=np.int32)
        mouth_polygon = sort_points_clockwise(mouth_polygon) # เรียง polygon ปากตามเข็มนาฬิกา
        
        # Find bounding box of face polygon
        x, y, w, h = cv2.boundingRect(face_polygon)
        # print("face rect = ",x,y,w,h)
        face_rect.append((x, y, w, h))  # Store bounding box coordinates
        
        # Fill the face polygon on the mask
        cv2.fillPoly(mask, [face_polygon], (255, 255, 255))
        cv2.fillPoly(mask, [L_eye_polygon], (0, 0, 0))
        cv2.fillPoly(mask, [R_eye_polygon], (0, 0, 0))
        cv2.fillPoly(mask, [mouth_polygon], (0, 0, 0))
        # Black out the area outside the face
        annotated_image = cv2.bitwise_and(annotated_image, mask)
        
        annotated_image = annotated_image[y:y + h, x:x + w]

    return annotated_image

def apply_clahe(src):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load the input image
    image = cv2.imread(src)
    image = resize_with_aspect_ratio(image,500)
    poster_image = cv2.pyrMeanShiftFiltering(image, 20,40,10)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rgb_imageP = cv2.cvtColor(poster_image, cv2.COLOR_BGR2RGB)

    def mass_clahe(im,mode):
        if mode == 1 :
            clahe = cv2.createCLAHE(clipLimit=5)
            image_bw = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            final_img = image_bw.copy()
            final_img = clahe.apply(image_bw) + 30
        elif mode == 2 :
            final_img = cv2.threshold(im, 30, 255, cv2.THRESH_BINARY)
        return final_img
    
    def find_contours(im):
        edged = cv2.Canny(im, 30, 200)
        contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours
    
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
            final1 = mass_clahe(draw_landmarks_on_image(rgb_image, results.multi_face_landmarks),1)
            _,final2 = mass_clahe(final1,2)
            finalP1 = mass_clahe(draw_landmarks_on_image(rgb_imageP, results.multi_face_landmarks),1)
            _,finalP2 = mass_clahe(finalP1,2)
            contours = find_contours(final2)
            contoursP = find_contours(finalP2)
            
            plt.figure(figsize=(10, 5))

            plt.subplot(2, 3, 1)
            plt.title("oily")
            plt.imshow(rgb_image)
            plt.axis('off')

            plt.subplot(2, 3, 2)
            plt.imshow(final1, cmap='grey')
            plt.axis('off')

            plt.subplot(2, 3, 3)
            plt.title(f"contours = {str(len(contours))}")
            plt.imshow(final2, cmap='grey')
            plt.axis('off')

            plt.subplot(2, 3, 4)
            plt.title("oily posterize")
            plt.imshow(rgb_imageP)
            plt.axis('off')

            plt.subplot(2, 3, 5)
            plt.imshow(finalP1, cmap='grey')
            plt.axis('off')

            plt.subplot(2, 3, 6)
            plt.title(f"contours = {str(len(contoursP))}")
            plt.imshow(finalP2, cmap='grey')
            plt.axis('off')

            plt.tight_layout()
            plt.show()
            
        else:
            print("No face landmarks detected")
            
apply_clahe('levle0_211.jpg')