from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

model = YOLO("yolov8m.pt")

results = model.predict("cat_dog.jpg")

result = results[0]

len(result.boxes)

# Create a black image
img = cv2.imread('cat_dog.jpg')
# Draw a diagonal blue line with thickness of 5 px
font = cv2.FONT_HERSHEY_SIMPLEX


for box in result.boxes:
  class_id = result.names[box.cls[0].item()]
  
  cords = box.xyxy[0].tolist()
  cords = [round(x) for x in cords]
  
  cv2.rectangle(img,(cords[0],cords[1]),(cords[2],cords[3]),(255,0,0),5)
  cv2.putText(img,class_id,(cords[0],cords[1]-20), font, 1,(255,0,0),2,cv2.LINE_AA)
  
  conf = round(box.conf[0].item(), 2)
  print("Object type:", class_id)
  print("Coordinates:", cords)
  print("Probability:", conf)
  print("---")

cv2.imshow('image',img)
if cv2.waitKey(0) == ord('q'):
    exit()