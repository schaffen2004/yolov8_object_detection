from ultralytics import YOLO
from PIL import Image

model = YOLO("yolov8m.pt")

results = model.predict("cat_dog.jpg")

result = results[0]

len(result.boxes)

for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  conf = round(box.conf[0].item(), 2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")
  
  Image.fromarray(result.plot()[:,:,::-1])