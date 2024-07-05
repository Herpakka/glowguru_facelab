import cv2
import numpy as np
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2

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

def posterize(image):
    spatial_radius = 20
    color_radius = 10
    num_levels = 40
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    # Apply pyrMeanShiftFiltering
    filtered_image = cv2.pyrMeanShiftFiltering(image, spatial_radius, color_radius)

    # Quantize the colors
    quantized_image = np.floor(filtered_image / (256 / num_levels)) * (256 / num_levels)
    quantized_image = np.uint8(quantized_image)
    quantized_image = cv2.cvtColor(quantized_image, cv2.COLOR_RGB2BGR)  # Convert to BGR

    return quantized_image

def sort_points_clockwise(points):
    # Calculate the centroid of the points
    centroid = np.mean(points, axis=0)
    # Sort points based on the angle with respect to the centroid
    sorted_points = sorted(points, key=lambda point: np.arctan2(point[1] - centroid[1], point[0] - centroid[0]))
    return np.array(sorted_points, dtype=np.int32)

def use_clahe(img,clahe):
    image_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image_bw_re = resize_with_aspect_ratio(image_bw, width=500)
    final_img = image_bw_re.copy()
    final_img = clahe.apply(image_bw_re) + 30 # default 30
    
    return final_img

def find_contours(img):
    edged = cv2.Canny(img, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len(contours)

def naming_code(og_name,allC_id,foreC_id,chkC_id):
    sp = "_"
    return og_name+"C"+allC_id+sp+foreC_id+sp+chkC_id

def draw_landmarks_Tzone(rgb_image, face_landmarks_list, mode):
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
        
        # T-zone polygon
        Tzone_indices = [152,148,176,140,32,194,182,181,167,45,51,3,196,122,193,55,65,52,53,63,68,54,103,67,109,10,338,297,332,284,298,293,283,282,295,285,417,351,419,248,281,275,393,405,406,418,262,369,400,377]
        Tzone_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in Tzone_indices], dtype=np.int32)
        
        # forehead polygon
        fore_indices = [55,65,52,53,63,68,54,103,67,109,10,338,297,332,284,298,293,283,282,295,285,8]
        fore_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in fore_indices], dtype=np.int32)
        
        
        # L cheek polygon
        Lchk_indices = [143,121,126,203,207,147]
        Lchk_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in Lchk_indices], dtype=np.int32)
        
        # L cheek polygon
        Rchk_indices = [372,350,355,423,427,376]
        Rchk_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in Rchk_indices], dtype=np.int32)
        
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
        mouth_indices = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]
        mouth_polygon = np.array([(int(face_landmarks.landmark[i].x * rgb_image.shape[1]), 
                                  int(face_landmarks.landmark[i].y * rgb_image.shape[0])) for i in mouth_indices], dtype=np.int32)
        mouth_polygon = sort_points_clockwise(mouth_polygon) # เรียง polygon ปากตามเข็มนาฬิกา
        
        # Find bounding box of face polygon
        x, y, w, h = cv2.boundingRect(face_polygon)
        face_rect.append((x, y, w, h))  # Store bounding box coordinates
        
        # Fill the face polygon on the mask
        if mode == "allface":
            cv2.fillPoly(mask, [face_polygon], (255, 255, 255))
            cv2.fillPoly(mask, [L_eye_polygon], (0, 0, 0))
            cv2.fillPoly(mask, [R_eye_polygon], (0, 0, 0))
            cv2.fillPoly(mask, [mouth_polygon], (0, 0, 0))
        elif mode == "Tzone":
            cv2.fillPoly(mask, [Tzone_polygon], (255,255,255))
            cv2.fillPoly(mask, [Rchk_polygon], (255,255,255))
            cv2.fillPoly(mask, [Lchk_polygon], (255,255,255))
            cv2.fillPoly(mask, [mouth_polygon], (0, 0, 0))
        elif mode == "fore":
            cv2.fillPoly(mask, [fore_polygon], (255, 255, 255))
        elif mode == "chk":
            cv2.fillPoly(mask, [Rchk_polygon], (255,255,255))
            cv2.fillPoly(mask, [Lchk_polygon], (255,255,255))
        
        # Black out the area outside the face
        annotated_image = cv2.bitwise_and(annotated_image, mask)
        
        annotated_image = annotated_image[y:y + h, x:x + w]

    return annotated_image
