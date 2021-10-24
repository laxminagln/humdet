# installation of required libraries
pip install opencv-python
pip install matplotlib
pip install deepface

# importing libraries
import cv2
import matplotlib.pyplot as plt
from deepface import DeepFace

# capturing picture and showing it
videoCaptureObject = cv2.VideoCapture(0)
ret, frame = videoCaptureObject.read()
color_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
plt.imshow(color_img)
videoCaptureObject.release()

# detects face
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = faceCascade.detectMultiScale(color_img, 1.1, 4)
for (x,y,u,v) in faces:
    cv2.rectangle(color_img, (x,y), (x+u,y+v), (0,0,225), 2)

# analyzes face and finds age, gender and expression
prediction = DeepFace.analyze(color_img)

# prints prediction
print(prediction)

# prints the details along with the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(color_img, prediction['dominant_emotion'], (150,200), font, 1, (225,225,225), 2, cv2.LINE_4)
cv2.putText(color_img, str(prediction['age']), (150,250), font, 1, (225,225,225), 2, cv2.LINE_4)
cv2.putText(color_img, prediction['gender'], (150,300), font, 1, (225,225,225), 2, cv2.LINE_4)
plt.imshow(color_img)
