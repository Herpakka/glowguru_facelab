import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def extract_lm(face_landmarks, indices, image_width, image_height):
    """
    Extracts the coordinates of specified landmarks from face_landmarks.

    Args:
    - face_landmarks: MediaPipe face landmarks object.
    - indices: List of landmark indices to extract.
    - image_width: Width of the image.
    - image_height: Height of the image.

    Returns:
    - List of (x, y) coordinates for each specified landmark.
    """
    landmarks = []
    for idx in indices:
        landmark = face_landmarks.landmark[idx]
        ax = landmark.x * image_width
        ay = landmark.y * image_height
        landmarks.append((ax, ay))
    return landmarks

def detect_tzone(image_path):
    # Load image
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Initialize Face Mesh model
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5) as face_mesh:
        results = face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("No face detected")
            return

        annotated_image = image.copy()

        for face_landmarks in results.multi_face_landmarks:
            # Draw face landmarks
            mp_drawing.draw_landmarks(
                image=annotated_image,
                landmark_list=face_landmarks,
                # connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
                )

            # Extract T-zone landmarks (forehead, nose, and chin)
            tzone_indices = tzone  # Example indices for forehead, nose, and chin
            ih, iw, _ = annotated_image.shape
            tzone_points = extract_lm(face_landmarks, tzone_indices, iw, ih)
            lcheek_points = extract_lm(face_landmarks, l_cheek, iw, ih)
            rcheek_points = extract_lm(face_landmarks, r_cheek, iw, ih)

            # Draw T-zone polygon
            tzone_points_np = np.array(tzone_points, np.int32)
            lcheek_points_np = np.array(lcheek_points, np.int32)
            rcheek_points_np = np.array(rcheek_points, np.int32)
            cv2.polylines(annotated_image, [tzone_points_np], isClosed=True, color=(0, 255, 0), thickness=2)
            cv2.polylines(annotated_image, [lcheek_points_np], isClosed=True, color=(0, 0, 255), thickness=2)
            cv2.polylines(annotated_image, [rcheek_points_np], isClosed=True, color=(0, 0, 255), thickness=2)

        # Display the result
        cv2.imshow('T-zone Detection', annotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        # Optionally save the annotated image
        cv2.imwrite('annotated_image.jpg', annotated_image)

# Example usage
tzone = [152,148,176,140,32,194,182,181,167,45,51,3,196,122,193,55,65,52,53,63,68,54,103,67,109,10,338,297,332,284,298,293,283,282,295,285,417,351,419,248,281,275,393,405,406,418,262,369,400,377]
l_cheek = [143,121,126,203,207,147]
r_cheek = [372,350,355,423,427,376]
detect_tzone('output/dry/9.jpg')
