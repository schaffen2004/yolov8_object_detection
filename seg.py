from ultralytics import YOLO
import cv2

model = YOLO("yolov8n-seg.pt")

path = "1.mp4"

vid = cv2.VideoCapture(path)

while vid.isOpened():
    ret, frame = vid.read()
    
    if ret:
        results = model(frame)
        
        req = results[0].plot()
        
        cv2.imshow('results',req)
        
        if cv2.waitKey(40) & 0xFF == ord('q'):
            break
    else:
        print('Error')
        break
vid.release()
cv2.destroyAllWindows()