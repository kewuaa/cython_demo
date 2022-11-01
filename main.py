# -*- coding: utf-8 -*-
#******************************************************************#
#
#                     Filename: main.py
#
#                       Author: kewuaa
#                      Created: 2022-05-23 12:32:47
#                last modified: 2022-06-07 23:25:31
#******************************************************************#
import pyimg
import cv2
import numpy as np
# img = cv2.imread('/mnt/e/PY/cv/project/apex_spotting/test.jpg')[..., 0]
img = cv2.imread('/mnt/c/Users/Lenovo/Desktop/rice.bmp')[..., 0][:512, :512]
# img = np.random.randint(33, size=(10, 10))
# img2 = pyimg.imgfilter(img, 3, 3, filter_type=pyimg.FILTER_MIN)
# print(img, img2, sep='\n\n\n')
img2 = pyimg.gaussian_filter(img, 3)
img2 = pyimg.morphology(img2, 50, 50, flag=pyimg.MORPH_TOPHAT, general=True)
img2 = pyimg.thresholding(img2, pyimg.THRESHOLD_OTSU)
# img2 = pyimg.canny(img2, 50, 100)
cv2.imshow('img2', img2.astype(np.uint8))
cv2.waitKey(0)

