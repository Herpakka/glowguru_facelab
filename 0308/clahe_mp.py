import cv2
import mediapipe as mp

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

# Initialize MediaPipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

# Load the image
image_path = 'svm/train/dry/28.jpg'
image = cv2.imread(image_path)
image = resize_with_aspect_ratio(image, width=500)

# Convert the BGR image to RGB
rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process the image and detect faces
with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
    results = face_detection.process(rgb_image)
    
# Print face locations
face_locations = []
if results.detections:
    for detection in results.detections:
        # Extract bounding box information
        bboxC = detection.location_data.relative_bounding_box
        ih, iw, _ = image.shape
        bbox = int(bboxC.xmin * iw), int(bboxC.ymin * ih), \
               int(bboxC.width * iw), int(bboxC.height * ih)
        face_locations.append(bbox)

# Print the face locations array
clahe = cv2.createCLAHE(clipLimit=5) #def = 5
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
final_img = image_bw.copy()
print(face_locations)
# x,y,w,h = list(face_locations)
print(type(face_locations))
for (x, y, w, h) in face_locations:
    face_roi = image_bw[y:y + h, x:x + w]  # Region of interest (face area) in grayscale
    final_img = clahe.apply(face_roi) + 30  # Apply CLAHE to the face region

# Ordinary thresholding the same image
    _, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

    # Threshold again
    _, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)

    # Dilation CLAHE image
    # img_erotion = cv2.erode(final_img2, kernel, iterations=1)

    # binary to rgb
    final_img2C = cv2.cvtColor(final_img2, cv2.COLOR_GRAY2RGB)

    # Draw contours
    edged = cv2.Canny(final_img2, 30, 200)
    contours, hierarchy = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print("Number of Contours found = " + str(len(contours)))
    cv2.drawContours(final_img2C, contours, -1, (0, 255, 0), 1)

    # resize again
    final_img = cv2.resize(final_img, (500, 500))
    final_img2C = cv2.resize(final_img2C, (500, 500))

    # Showing images
    cv2.imshow("og",image)
    cv2.imshow("threshold after CLAHE ,contours:" + str(len(contours)), final_img2C)
    # cv2.imshow("ordinary threshold", ordinary_img)
    cv2.imshow("CLAHE image", final_img)
    # cv2.imshow("Erode image", img_erotion)
    cv2.waitKey(0)