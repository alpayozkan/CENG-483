import numpy as np
import cv2


convolution_kernel = np.array([[0, 1, 0], 
                               [1, 1.5, 1], 
                               [0, 1, 0]])

sharpen = np.array([[0, -1, 0], 
                    [-1, 5, -1], 
                    [0, -1, 0]])

laplacian = np.array([[0, 1, 0], 
                      [1, -4, 1], 
                      [0, 1, 0]])

emboss = np.array([[-2, -1, 0], 
                   [-1, 1, 1], 
                   [0, 1, 2]])

outline = np.array([[-1, -1, -1], 
                    [-1, 8, -1], 
                    [-1, -1, -1]])

bottom_sobel = np.array([[-1, -2, -1], 
                         [0, 0, 0], 
                         [1, 2, 1]])

left_sobel = np.array([[1, 0, -1], 
                       [2, 0, -2], 
                       [1, 0, -1]])

right_sobel = np.array([[-1, 0, 1], 
                        [-2, 0, 2], 
                        [-1, 0, 1]])

top_sobel = np.array([[1, 2, 1], 
                      [0, 0, 0], 
                      [-1, -2, -1]])

shift_right = np.array([[0, 0, 0], 
                      [1, 0, 0], 
                      [0, 0, 0]])

# animal.png de hata veriyor
# libpng warning: iCCP: known incorrect sRGB profile

def applyKernel(img, kernel, n):
    for i in range(n):
        img = cv2.filter2D(img, -1, kernel)
    return img

image = cv2.imread('images/flower.png')
result = applyKernel(image, shift_right, 40)

cv2.imshow('frame1', result)
cv2.imshow('frame2', image)
cv2.waitKey(0)

