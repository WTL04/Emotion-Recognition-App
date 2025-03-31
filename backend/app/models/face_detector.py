import cv2
import mediapipe as mp # pretrained face detection model 

# open default webcam
cam = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# loop capturing webcam frames
while True:
    ret, frame = cam.read()
    
    # display captured frame
    cv2.imshow("Camera", frame)

    # press 'q' to exit loop 
    if cv2.waitKey(1) == ord('q'):
        break

# release capture and destory all windows
cam.release()
cv2.destoryAllWindows()
