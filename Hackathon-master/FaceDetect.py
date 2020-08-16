import cv2
import sys
from PIL import Image
import gpiozero


def CatchUsbVideo(window_name, camera_idx):
    cv2.namedWindow(window_name)
    
    cap = cv2.VideoCapture(camera_idx)
    
    classifier = cv2.CascadeClassifier("/opt/anaconda3/lib/python3.8/site-packages/cv2/data/haarcascade_frontalface_alt2.xml")
    color = (0,255,0)
    mortor = Mortor(17,18)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        faceRects = classifier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32,32))
        
        if len(faceRects) >0:
            for faceRect in faceRects:
                x,y,w,h = faceRect
                cv2.rectangle(frame, (x-10,y-10),(x+w+10,y+h+10),color,2)
            Mortor.backward()
        else:
            reverse()
            
            
        
        cv2.imshow(window_name, frame)
        c= cv2.waitKey(10)
        if c &0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

CatchUsbVideo("Recognize Face Area",0)
    
