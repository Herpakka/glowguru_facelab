import cv2
import numpy as np

# Reading the image from the present directory
image = cv2.imread("006.jpg")

# Load the face detection XML
cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)

# Resizing the image for compatibility
image = cv2.resize(image, (500, 600))

# Convert the image to grayscale
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = faceCascade.detectMultiScale(image_bw)

# Declare CLAHE
clahe = cv2.createCLAHE(clipLimit=5)  # clipLimit=5 is the default value

# Create a copy of the original image to apply CLAHE to the faces only
final_img = image.copy()

# Loop through detected faces and apply CLAHE
for (x, y, w, h) in faces:
    face_roi = image_bw[y:y+h, x:x+w]  # Region of interest (face area) in grayscale
    face_clahe = clahe.apply(face_roi)  # Apply CLAHE to the face region
    face_clahe = np.clip(face_clahe + 30, 0, 255).astype(np.uint8)  # Add 30 to the pixel values and clip

    # Replace the face region in the color image with the CLAHE applied version
    final_img[y:y+h, x:x+w] = cv2.cvtColor(face_clahe, cv2.COLOR_GRAY2BGR)

# Display the final image
cv2.imshow("Image with CLAHE on Faces", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the final image
cv2.imwrite("final_image.jpg", final_img)
