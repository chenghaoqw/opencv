# -*- coding: cp936 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
videoCapture = cv2.VideoCapture('test.avi')
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)

size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
while(1):
 ret,frame=videoCapture.read()
 if frame is not None:
  hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

  lower_blue=np.array([0,43,46])
  upper_blue=np.array([10,255,255])

  mask=cv2.inRange(hsv,lower_blue,upper_blue)

  res=cv2.bitwise_and(frame,frame,mask=mask)
  contours, heirs = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  contours=sorted(contours, key=lambda cont : cv2.contourArea(cont),reverse=True)
  cv2.drawContours(frame, contours, 0, (0, 0, 255), 1)

  # 计算外接圆的数据
  (x, y), radius = cv2.minEnclosingCircle(contours[0])
  center = (int(x), int(y))
  radius = int(radius)

  surf = cv2.SURF(100000)
  kps, des = surf.detectAndCompute(mask, None)

  blob=cv2.drawKeypoints(frame, kps, None, (0, 255, 255), 4)
  # cv2.imshow('blob', blob)
  # 绘制所有斑点


  kp_list=[];
  for kp in kps:
    kp_list.append(kp)
  kp_list=sorted(kp_list,key=lambda kp : (int(kp.pt[0])-int(x))^2+(int(kp.pt[1])-int(y))^2,reverse=True)
  if (not (len(kp_list)==0) ):
    number = (int(kp_list[0].pt[0]), int(kp_list[0].pt[1]))
    distance=(int(kp_list[0].pt[0])-int(x))^2 + (int(kp_list[0].pt[1])-int(y))^2
    if distance <50:
     cv2.line(blob, number, number, (0, 255, 0), 3)
  print str(mask[center]) + " " + str(len(kps)) + " "+ str(center)
  # cv2.imshow('mask',mask)
  # cv2.imshow('res',res)
  # cv2.imshow('edge',edges)
  cv2.imshow('frame', blob)
  # cv2.imshow('blob',blob)
  k=cv2.waitKey(5)&0xFF
  if k==27:
   break
 else:
     break

cv2.destroyAllWindows()

