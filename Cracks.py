from matplotlib import pyplot as plt
import numpy as np
import imageio
import cv2

%matplotlib inline

crack_cascade = cv2.CascadeClassifier('CRACKS.xml')

img = cv2.imread('013.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cracks = crack_cascade.detectMultiScale(gray,1.5,3)

for (x,y,w,h) in cracks:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('img',img)

    
k = cv2.waitKey(0)

if k == 27 & ord("q"):
    cv2.destroyAllWindows()