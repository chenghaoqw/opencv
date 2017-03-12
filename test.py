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
  # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # �Ҷȱ任
  # cv2.imshow('�Ҷ�ͼ��', gray)
  # blur = cv2.GaussianBlur(gray, (5, 5), 0)  # ��˹�˲�
  # edge = cv2.Canny(blur, 100, 200, True)  # ��canny���ӱ�Ե���
  #
  # # ʹ��3*3���ں˶��ݶ�ͼ����ƽ��ģ����������ƽ����ͼ������ͼ���еĸ�Ƶ����,�˲��費����
  # blured = cv2.blur(edge, (3, 3))
  # # ��ֵ��������OTSU
  # ret, binary = cv2.threshold(blured, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
  # cv2.imshow('��ֵ��ͼ��', binary)
  # # print binary[10,:]
  # # ��̬ѧ����
  # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
  # closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
  # # �ȸ�ʴ������ȥ�������İ�ɫ������
  # # closing=cv2.erode(closing,kernel,iterations=6)
  # # closing=cv2.dilate(closing,kernel,iterations=6)
  # cv2.imshow('��̬ѧ����ͼ��', closing)
  # # ȥ��α���ƣ�ͨ���������
  # contours, hierarchy = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  # cv2.drawContours(ret, contours, 2, (0, 255, 255), 10)
  # cv2.imshow('����', ret)
  # print contours
  k=cv2.waitKey(5)&0xFF
  if k==27:
   break
 else:
     break

cv2.destroyAllWindows()

