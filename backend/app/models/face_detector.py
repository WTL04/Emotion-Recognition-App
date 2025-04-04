import cv2
import torch
import pandas as pd 
import numpy as np

emotion_labels = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"]

model = torch.load("emotion_model_full.pth", map_location = torch.device('cpu'), weights_only = False) # load saved trained model
model.eval()

# loading pre-trained classifer
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# open default webcam
cam = cv2.VideoCapture(0)
if not cam.isOpened():
    raise IOError("Cannot open webcam")

# loop capturing webcam frames
while True:

    # ret: bool indicating if frame was captured successfully
    # frame: the actual image frame being captured
    ret, frame = cam.read()
    if not ret:
        break
    
    # convert frame to grayscale since Haar Cascades works better with grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

     # Draw bounding boxes around faces
    for (x, y, w, h) in faces:

        # draw green binding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
   
    cv2.imshow("YOLOv8 Tracking", frame)

    # press 'q' to exit loop 
    if cv2.waitKey(1) == ord('q'):
        break

# release capture and destory all windows
cam.release()
cv2.destroyAllWindows()
