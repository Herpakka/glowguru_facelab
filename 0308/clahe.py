import cv2
import numpy as np

image = cv2.imread("4.jpg") # Reading the image from the present directory

cascPath = "haarcascade_frontalface_default.xml" # face detection xml
faceCascade = cv2.CascadeClassifier(cascPath) # รับ path ไฟล์ของไฟล์ xml ที่เก็บ haar cascades ที่ใช้ระบุใบหน้า

# Resizing the image for compatibility
image = cv2.resize(image, (500, 600))

kernel = np.ones((5, 5), np.uint8)

# The initial processing of the image
# image = cv2.medianBlur(image, 3)
image_bw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

faces = faceCascade.detectMultiScale(image_bw) # ตรวจหาหน้า

# The declaration of CLAHE
# clipLimit -> Threshold for contrast limiting
clahe = cv2.createCLAHE(clipLimit=5) #def = 5
final_img = clahe.apply(image_bw) + 30

# Ordinary thresholding the same image
_, ordinary_img = cv2.threshold(image_bw, 155, 255, cv2.THRESH_BINARY)

# Threshold again
_, final_img2 = cv2.threshold(final_img, 30, 255, cv2.THRESH_BINARY)

# Dilation CLAHE image
# img_erotion = cv2.erode(final_img2, kernel, iterations=1)

# binary to rgb
final_imgC = cv2.cvtColor(final_img, cv2.COLOR_GRAY2RGB)
final_img2C = cv2.cvtColor(final_img2, cv2.COLOR_GRAY2RGB)

# Draw face contour
try:
    print("face loc:",faces)
    x,y,w,h = faces[0]
    cv2.rectangle(final_imgC, (x, y), (x+w, y+h), (0, 0, 255), 2)
except:
    print("face out of bound")

# Draw contours
edged = cv2.Canny(final_img2, 30, 200)
contours, hierarchy  = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print("Number of Contours found = " + str(len(contours)))
cv2.drawContours(final_img2C, contours, -1, (0, 255, 0), 1)

# Showing the two images
cv2.imshow("threshold after CLAHE ,contours:" + str(len(contours)), final_img2C)
# cv2.imshow("ordinary threshold", ordinary_img)
cv2.imshow("CLAHE image", final_imgC)
# cv2.imshow("Erode image", img_erotion)
cv2.waitKey(0)
