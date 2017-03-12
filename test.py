# -*- coding: cp936 -*-
import numpy as np
import cv2
from matplotlib import pyplot as plt
videoCapture = cv2.VideoCapture('5.avi')
fps = videoCapture.get(cv2.cv.CV_CAP_PROP_FPS)

size = (int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)),
        int(videoCapture.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)))
while(1):
 ret,frame=videoCapture.read()
 if frame is not None:
  hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

  lower_blue=np.array([0,50,50])
  upper_blue=np.array([180,255,255])


  mask=cv2.inRange(hsv,lower_blue,upper_blue)

  res=cv2.bitwise_and(frame,frame,mask=mask)
  contours, heirs = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  img = frame
  edges = cv2.Canny(img, 100, 200)

  cv2.imshow('frame',frame)
  cv2.imshow('mask',mask)

  cv2.imshow('res',res)
  cv2.imshow('edge',edges)
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # 灰度变换
  # cv2.imshow('灰度图像', gray)
  # blur = cv2.GaussianBlur(gray, (5, 5), 0)  # 高斯滤波
  # edge = cv2.Canny(blur, 100, 200, True)  # 用canny算子边缘检测
  #
  # # 使用3*3的内核对梯度图进行平均模糊，有助于平滑地图表征的图形中的高频噪声,此步骤不能少
  # blured = cv2.blur(edge, (3, 3))
  # # 二值化，采用OTSU
  # ret, binary = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # cv2.imshow('二值化图像', binary)
  # # print binary[10,:]
  # # 形态学处理
  # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
  # closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  # # 先腐蚀再膨胀去除孤立的白色点噪声
  # # closing=cv2.erode(closing,kernel,iterations=6)
  # # closing=cv2.dilate(closing,kernel,iterations=6)
  # cv2.imshow('形态学运算图像', closing)
  # # 去除伪车牌，通过长宽比例
  # contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.drawContours(ret, contours, 2, (0, 255, 255), 10)
  # cv2.imshow('轮廓', ret)
  # print contours
  k=cv2.waitKey(5)&0xFF
  if k==27:
   break
 else:
     break

cv2.destroyAllWindows()

