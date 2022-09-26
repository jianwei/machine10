# -*- coding: utf-8 -*-  

import os
import cv2
import numpy as np



def handl_Canny(pic_path='img/1.jpg'):
    #读取图像
    img=cv2.imread(pic_path)
    #缩放图像的大小
    img=cv2.resize(src=img,dsize=(512,512))
    #对图像进行边缘检测，低阈值=50 高阈值=200
    # img_canny=cv2.Canny(img,threshold1=100,threshold2=100)
    #显示检测之后的图像
    # cv2.imshow('image',img_canny)
    #显示原图像
    cv2.imshow('src',img)

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # _, img_binary = cv2.threshold(src=img_gray, thresh=50, maxval=255,
    #                         type=cv2.THRESH_BINARY)
    # cv2.imshow('image_binary', img_binary)


    # 指数变换
    # gamma_img = 10 * np.power(img, 0.3)
    # # 截断，把大于255的像素值变为255
    # gamma_img[gamma_img>255] = 255    
    # # 像素值变为整数
    # gamma_img = np.asarray(gamma_img, np.uint8) 

    # cv2.imshow('gamma_image', gamma_img)

    # # 线性变换
    linear_img = img - 150
    linear_img.max()    # 最大值364.0
    # 截断，把大于255的像素值变为255
    linear_img[linear_img>255] = 255
    linear_img = np.asarray(linear_img, np.uint8) # 像素值变为整数
    cv2.imshow('linear image', linear_img)
    
    img_cat_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    cv2.imshow("hsv", img_cat_hsv)




    # blur = cv2.GaussianBlur(linear_img,(15,15),0)
    # cv2.imshow('linear image', blur)

    # img_blur = cv2.Canny(blur,100,175)
    # cv2.imshow('img_blur1', img_blur)



    # img_blur = cv2.Canny(linear_img,50,175)
    # cv2.imshow('img_blur1', img_blur)

    # img_blur = cv2.Canny(img,125,175)
    # cv2.imshow('img_blur2', img_blur)


    # img_gray = cv2.cvtColor(linear_img, cv2.COLOR_BGR2GRAY)    # 转为灰度图
    # img_hist = cv2.equalizeHist(img_gray)   # 对单通道图像进行均衡化
    # cv2.imshow('gray', img_hist)





    key = cv2.waitKey(0)
    if(ord(key)=="q" ):
        exit()
    cv2.destroyAllWindows()

#销毁所有的窗口
cv2.destroyAllWindows()
if __name__ == '__main__':
    # print('PyCharm')
    handl_Canny("img/1.jpg")
    # handl_Canny("img/2.jpg")
    # handl_Canny("img/3.jpg")
    
