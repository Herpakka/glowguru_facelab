import cv2
import numpy as np

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

image = cv2.imread("svm/train/dry/58.jpg") # Reading the image from the present directory

cascFPath = "clahe/haarcascade_frontalface_default.xml" # face detection xml
cascEPath = "clahe/haarcascade_eye.xml"
# cascMPath = "mouth.xml"
faceCascade = cv2.CascadeClassifier(cascFPath) # รับ path ไฟล์ของไฟล์ xml ที่เก็บ haar cascades ที่ใช้ระบุใบหน้า
eyeCascade = cv2.CascadeClassifier(cascEPath)
# mouthCascade = cv2.CascadeClassifier(cascMPath)

# Resizing the image for compatibility
image = resize_with_aspect_ratio(image, width=500)

kernel = np.ones((5, 5), np.uint8)

# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(image_bw) # ตรวจหาหน้า
eyes = eyeCascade.detectMultiScale(image_bw)

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit=5) #def = 5
# final_img = clahe.apply(image_bw) + 30
final_img = image_bw.copy()

try:
    x,y,w,h = faces[0]
    ex,ey,ew,eh = eyes[0]
except:
    print("face out of bound")
else:
    print("eye loc:", eyes)
    print("face loc:", faces)
    # Loop through detected faces and apply CLAHE
    for (x, y, w, h) in faces:
        face_roi = image_bw[y:y + h, x:x + w]  # Region of interest (face area) in grayscale
        final_img = clahe.apply(face_roi) + 30  # Apply CLAHE to the face region
        
        eyes = eyeCascade.detectMultiScale(face_roi)
        for (ex,ey,ew,eh) in eyes :
            cv2.rectangle(final_img,(ex,ey),(ex+ew,ey+eh),(255,255,255),-1)
        # mouth = mouthCascade.detectMultiScale(face_roi)
        # for (mx,my,mw,mh) in mouth :
        #     cv2.rectangle(final_img,(mx,my),(mx+mw,my+mh),(255,255,255),-1)
        

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