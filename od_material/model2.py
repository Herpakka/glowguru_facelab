from ultralytics import YOLOv10

model_path = 'od_material/TFAPI_Face_Detection/best.pt'
model = YOLOv10(model_path)
results = model(source='/content/X-Ray-Baggage-3/test/images', conf=0.25,save=True)