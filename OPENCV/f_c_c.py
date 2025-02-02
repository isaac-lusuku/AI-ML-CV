# MORE COMPUTER VISION AND IMAGE PROCESSING WITH OPENCV

import cv2 as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# importing the images that we are to work with
image1 = cv.imread("Assets/soccer_practice.jpg")  # soccer
image2 = cv.imread("Assets/cat1.jpg")  # cat

# resizing the images to smaller sizes
image1 = cv.resize(image1, (0, 0), fx=0.4, fy=0.4)
image2 = cv.resize(image2, (0, 0), fx=0.3, fy=0.3)

# displaying the original images
# cv.imshow("original_img1", image1)
# cv.imshow("original_img2", image2)

"""--->IMAGE SMOOTHING AND BLURRING<---"""
# # averaging
# avg_blur = cv.blur(image1, (7, 7))
# cv.imshow("avg_blur", avg_blur)
#
# # gaussian blurring
# gsn_blur = cv.GaussianBlur(image1, (7, 7), 0)
# cv.imshow("gsn_blur", gsn_blur)
#
# # median blur
# mdn_blur = cv.medianBlur(image1, 7)
# cv.imshow("mdn_blur", mdn_blur)
#
# # bilateral filter
# blt_blur = cv.bilateralFilter(image1, 15, 75, 75)
# cv.imshow("blt_blur", blt_blur)
#
# cv.waitKey(0)
# cv.destroyAllWindows()

"""-->EDGE DETECTION<---"""
# # CANNY
# # Turning the image grayscale and blurring it
# image2 = cv.cvtColor(image2, cv.COLOR_BGR2GRAY)
# gsn_blur = cv.GaussianBlur(image2, (3, 3), 0)
# cv.imshow("gsn_blur", gsn_blur)
#
# # applying the canny
# canny_image = cv.Canny(gsn_blur, 100, 200)
# cv.imshow("CANNY", canny_image)
#
# # SOBEL EDGE DETECTION
# sobelXY = cv2.Sobel(gsn_blur, cv.CV_64F, 1, 1)
# cv.imshow("SOBEL", sobelXY)
#
# cv.waitKey(0)
# cv.destroyAllWindows()


"""--->HISTOGRAM CALCULATING AND EQUALIZATION<---"""
# HISTOGRAM CALCULATION
# B, G, R = cv.split(image2)  # splitting the image into color channels
#
# # calculating the histograms
# hist_B = cv.calcHist([B], [0], None, [256], [0, 256])
# hist_G = cv.calcHist([G], [0], None, [256], [0, 256])
# hist_R = cv.calcHist([R], [0], None, [256], [0, 256])
#
# # hist visualization
# plt.subplot(2, 2, 1)
# plt.plot(hist_B, "BLUE")
# plt.subplot(2, 2, 2)
# plt.plot(hist_G, "GREEN")
# plt.subplot(2, 2, 3)
# plt.plot(hist_R, "RED")
#
# plt.show()
#
# # HISTOGRAM EQUALIZATION
# eq_B = cv.equalizeHist(B)
# eq_G = cv.equalizeHist(G)
# eq_R = cv.equalizeHist(R)
#
# # merging the equalized
# merged = cv.merge([eq_B, eq_G, eq_R])
#
# # displaying the images
# cv.imshow("original_img2", image2)
# cv.imshow("merged channels", merged)
#
# cv.waitKey(0)
# cv.destroyAllWindows()





