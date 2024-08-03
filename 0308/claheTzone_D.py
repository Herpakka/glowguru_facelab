# use object detection model to detect dense/spread clahe pixel

import cv2
import supervision as sv
from ultralytics import YOLO
from helper import resize_with_aspect_ratio

# Load the YOLO model
model = YOLO("od_material/TFAPI_Face_Detection/XML_clahe_yolov8n.pt")

# Read the image
image_path = "clahe/fore2exam.jpg"  # Replace with the path to your image
image = cv2.imread(image_path)
image = resize_with_aspect_ratio(image,width=500)

# Run the YOLO model on the image
results = model(image)[0]

# Convert YOLO results to Supervision detections
detections = sv.Detections.from_ultralytics(results)

# Create annotators for bounding boxes and labels
box_annotator = sv.BoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Generate labels for the detections
labels = [
    f"{class_name}"
    for class_name
    in zip(detections['class_name'])
]

# Annotate the image with bounding boxes
annotated_image = box_annotator.annotate(
    scene=image.copy(), detections=detections)

# Annotate the image with labels
annotated_image = label_annotator.annotate(
    scene=annotated_image, detections=detections, labels=labels)

sv_loc = detections.xyxy
for i, label in enumerate(labels):
    print(f'Bounding Box: {sv_loc[i]}, Label: {label}')
    
# Display the annotated image
sv.plot_image(annotated_image)
