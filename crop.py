import cv2
import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import glob
from skimage import io

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

        # Find bounding box of face polygon
        x, y, w, h = cv2.boundingRect(face_polygon)
        # print("face rect = ",x,y,w,h)
        face_rect.append((x, y, w, h))  # Store bounding box coordinates
        
        annotated_image = annotated_image[y:y + h, x:x + w]

    return annotated_image

def apply_clahe(src,fold):
    mp_face_mesh = mp.solutions.face_mesh
    
    # Load the input image
    # print("type = ",type(src))
    src = src.replace("\\","/")
    image = io.imread(src,0)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    try :
        # Initialize the FaceMesh model
        with mp_face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5) as face_mesh:
            if mp_face_mesh :
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
                    osrc_list = list(src.split("/"))
                    osrc =  "coutput/"+fold+"/"+osrc_list[3]

                    try :
                        cv2.imwrite(osrc,final_img2)
                        print("write : ",osrc," complete")
                    except Exception as e:
                        print(e)
                    
                else:
                    print("No face landmarks detected")
            else :
                osrc_list = list(src.split("/"))
                osrc =  "coutput/"+folder+"/"+osrc_list[3]
                
                cv2.imwrite(osrc,image)
                print("No face landmarks detected")
                print("write : ",osrc," complete")
    except Exception as e:
        print(e)
            
            
src0 = "Classification/Classification/JPEGImages/levle0_2.jpg"
src1 = "dataset acne04-1200/dataset acne04-1200/test/01F3MMX1JKX31TSGA6GK7S16T1_jpeg.rf.ad689bc304ad18e64b556dee9d129f03.jpg"
anyf = "\\*\\*\\*.*"
path = ["Classification-20240626T083635Z-001"+anyf,"dataset acne04-1200"+anyf,"dataset acne04-1200"+anyf]
folder = ["class","1200","1400"]
for src in glob.glob(path[0]):
    apply_clahe(src,folder[0])
for src in glob.glob(path[1]):
    apply_clahe(src,folder[1])
for src in glob.glob(path[2]):
    apply_clahe(src,folder[2])
