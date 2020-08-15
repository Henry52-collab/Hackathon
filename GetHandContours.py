#!/usr/bin/env python
# coding: utf-8


import numpy as np
import cv2
import math

#img = cv2.imread("/Users/wanghuiyuan/Desktop/ffff.png",cv2.IMREAD_COLOR)

#cv2.imshow("image",img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()




import numpy as np
import cv2

class Capture(object):
    
    def __init__(self,deviceID=0):
        self.deviceID = deviceID
        self.capture = cv2.VideoCapture(self.deviceID)
    
    def read(self):
        _,frame = self.capture. read()
        frame = cv2.bilateralFilter(frame, 9,75, 75)
        image = Image.fromarray(frame)
        return image
    
    




def _remove_background(frame):
    fgbg = cv2.createBackgroundSubtractorMOG2()
    
    fgmask= fgbg.apply(frame)
    
    kernel = np.ones((3,3),np.uint8)
    fgmask = cv2.erode(fgmask, kernel, iterations=1)
    res= cv2.bitwise_and(frame, frame, mask=fgmask)
    kernel = np.ones((3,3), np.uint8)
    erosion= cv2.erode(res,kernel)
    dilation = cv2.dilate(erosion, kernel)
    return dilation




def _bodyskin_detect(frame):
    
    ycrcb= cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    (_, cr, _) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr,(5,5),0)
    _, skin = cv2.threshold(cr1, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("image1",skin)
    
    return skin

def _remove_noise(frame):
    blur= cv2.blur(frame, (3,3))
    blur = cv2.bilateralFilter(blur, 9, 1,1)
    return blur




from enum import Enum

Event=Enum('Event',('NONE','ONE','TWO','THREE','FOUR'))

class event(object):
    
    def __init__ (self, eventtype=Event.NONE):
        self.type= eventtype
    
    def setType(self,newType):
        self.type =newType
    

   

    




from collections import namedtuple

KeyCode= namedtuple('KeyCode',['ESCAPE','Q','q'])

Keycode = KeyCode._make([27,81,113])


COLOR = namedtuple('COLOR',['RED','GREEN','BLUE'])
Color = COLOR._make([(0,0,255),(0,255,0),(255,0,0)])


import cv2


#hull = cv2.convexHull(cnt,returnPoints= False)
#defects = cv2.convexityDefects(cnt,hull)


def _get_contours(array):
    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(array,cv2.MORPH_OPEN,kernel )
    closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE,kernel)
    contours,h = cv2.findContours(closed, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return contours

def _get_eucledian_distance(beg, end):#计算两点之间的坐标
    i=str(beg).split(',')
    j=i[0].split('(')
    x1=int(j[1])
    k=i[1].split(')')
    y1=int(k[0])
    i=str(end).split(',')
    j=i[0].split('(')
    x2=int(j[1])
    k=i[1].split(')')
    y2=int(k[0])
    d=math.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))
    return d

def _get_defects_count(array, contour, defects, verbose = False):
    ndefects=0
    
    for i in range(defects.shape[0]):
        s,e,f,_=defects[i,0]
        beg     = tuple(contour[s][0])
        end     = tuple(contour[e][0])
        far     = tuple(contour[f][0])
        a       = _get_eucledian_distance(beg, end)
        b       = _get_eucledian_distance(beg, far)
        c       = _get_eucledian_distance(end, far)
        angle   = math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) # * 57
        
        if angle <= math.pi/2:
            ndefects = ndefects +1
            
            if verbose:
                cv2.circle(array, far,3, Color.RED, -1)
        if verbose:
            cv2.line(array, beg,end, Color.RED, 1)
    return array, ndefects


def grdetect(array, verbose=False):
    myevent = event(Event.NONE)
    copy = array.copy()
    array= _remove_background(array)
    thresh = _bodyskin_detect(array)
    
    contours = _get_contours(thresh.copy())
    largecont = max(contours, key= lambda contour:cv2.contourArea(contour))
    
    hull = cv2.convexHull(largecont, returnPoints=False)
    defects = cv2.convexityDefects(largecont, hull)
    
    if defects is not None:
        
        copy, ndefects = _get_defects_count(copy, largecont, defects, verbose=verbose)
        
        if   ndefects == 0:
            myevent.setType(Event.NONE)
        elif ndefects == 1:
            myevent.setType(Event.ONE)
        elif ndefects == 2:
            myevent.setType(Event.TWO)
        elif ndefects == 3:
            myevent.setType(Event.THREE)
        elif ndefects == 4:
            myevent.setType(Event.FOUR)
        return myevent
    
    


def HSVBin(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    
    lower_skin = np.array([100,50,0])
    upper_skin = np.array([125,255,255])
    
    mask = cv2.inRange(hsv, lower_skin,upper_skin)
    return mask



import time
import cv2
from PIL import Image


if __name__=='__main__':
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()) and (cv2.waitKey(10) not in [Keycode.ESCAPE, Keycode.Q, Keycode.q]):
        ret, frame = cap.read()
        frame = _remove_noise(frame)
#        image = Image.fromarray(frame)

        res= _remove_background(frame)

        mask = HSVBin(res)
        

        
        contours= _get_contours(mask)
        
        if grdetect(frame).type==Event.FOUR:
            print('444444444444444444444444444444444444')
        
        cv2.drawContours(frame, contours, -1 , (0,255,0),2)
        cv2.imshow('capture',frame)
    
    





