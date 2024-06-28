import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import glob
import multiprocessing as mp_pool
import os

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
        # print("face rect = ",x,y,w,h)
        face_rect.append((x, y, w, h))  # Store bounding box coordinates
        
        # Fill the face polygon on the mask
        cv2.fillPoly(mask, [face_polygon], (255, 255, 255))
        cv2.fillPoly(mask, [L_eye_polygon], (0, 0, 0))
        cv2.fillPoly(mask, [R_eye_polygon], (0, 0, 0))
        cv2.fillPoly(mask, [mouth_polygon], (0, 0, 0))
        # Black out the area outside the face
        annotated_image = cv2.bitwise_and(annotated_image, mask)

        # Crop the image to the face bounding box
        annotated_image = annotated_image[y:y + h, x:x + w]

    # Check if the face region is valid for further processing
    if (annotated_image.shape[1] >= 3 * annotated_image.shape[0]) or (annotated_image.shape[0] >= 4 * annotated_image.shape[1]):
        return rgb_image
    else:
        return annotated_image

def apply_clahe(src):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load the input image
    src = src.replace("\\","/")
    image = cv2.imread(src)
    poster_image = cv2.pyrMeanShiftFiltering(image, 20, 40, 10)
    rgb_image = cv2.cvtColor(poster_image, cv2.COLOR_BGR2RGB)
    
    try:
        # Initialize the FaceMesh model
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            if mp_face_mesh:
                # Process the image to get the face landmarks
                results = face_mesh.process(rgb_image)

                # Draw the landmarks on the image
                if results.multi_face_landmarks:
                    annotated_image = draw_landmarks_on_image(rgb_image, results.multi_face_landmarks)
                    crop_image = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
                    clahe = cv2.createCLAHE(clipLimit=5)
                    image_bw = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2GRAY)
                    final_img = clahe.apply(image_bw) + 30
                    _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)
                    edged = cv2.Canny(final_img2, 30, 200)
                    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    osrc_list = list(src.split("/"))
                    osrc = "poutput/" + osrc_list[1] + "/" + str(len(contours)) + "_" + osrc_list[2]

                    try:
                        cv2.imwrite(osrc, final_img2)
                        print("write : ", osrc, " complete")
                    except Exception as e:
                        print(e)
                else:
                    print("No face landmarks detected")
            else:
                print("No face landmarks detected")
    except Exception as e:
        print(e)

def process_images_in_folder(folder):
    for src in glob.glob(folder + "/*.*"):
        apply_clahe(src)

def main():
    folder = "train/*"
    subfolders = [f for f in glob.glob(folder)]

    # Use multiprocessing to process each subfolder
    with mp_pool.Pool(processes=os.cpu_count()) as pool:
        pool.map(process_images_in_folder, subfolders)

if __name__ == "__main__":
    main()
