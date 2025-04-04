import cv2
import torch
import torch.nn.functional as F
import pandas as pd 
import numpy as np

# initilaize array of emotion labels
emotion_labels = ["Angry", "Happy", "Neutral", "Sad"]

model = torch.load("emotion_model_full.pth", map_location = torch.device("cuda" if torch.cuda.is_available() else "cpu")) # load saved trained model
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

    for (x, y, w, h) in faces:
        
        # getting face region, resulting face tensor: [1, 1, 48, 48]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48)) # resize face to 48x48 pixels
        face = np.expand_dims(face, axis=0) # add batch dimension
        face = np.expand_dims(face, axis=0) # add channel dimension
        face = torch.tensor(face, dtype=torch.float32) / 255.0  # array -> tensor, normalize pixels

        with torch.no_grad():
            output = model(face)
            # softmax turns tensor into probabilities
            # argmax returns index of highest probability
            # .item() extracts value as integer from tensor
            prediction = torch.argmax(F.softmax(output, dim=1), dim=1).item()

        # get labeled emotion from 
        emotion = emotion_labels[prediction]
        
        # draw binding box and put emotion on screen
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX,0.9, (36, 255, 12), 2) 
   
    cv2.imshow("YOLOv8 Tracking", frame)

    # press 'q' to exit loop 
    if cv2.waitKey(1) == ord('q'):
        break

# release capture and destory all windows
cam.release()
cv2.destroyAllWindows()
