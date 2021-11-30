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

shift_left = np.array([[0, 0, 0], 
                      [0, 0, 1], 
                      [0, 0, 0]])

gauss = (1/16)*np.array([[1, 2, 1], 
                        [2, 4, 2], 
                        [1, 2, 1]], dtype=np.float64)

inverse_gauss = (1/25)*np.array([[4, 2, 4], 
                                [2, 1, 2], 
                                [4, 2, 4]], dtype=np.float64)

deriv_x = np.array([[0, 0, 0], 
                      [1, 0, -1], 
                      [0, 0, 0]], dtype=np.float64)

deriv_y = np.array([[0, 1, 0], 
                      [0, 0, 0], 
                      [0, -1, 0]], dtype=np.float64)

deriv_rd = np.array([[0, 0, 1], 
                      [0, 0, 0], 
                      [-1, 0, 0]], dtype=np.float64)

deriv_ld = np.array([[1, 0, 0], 
                      [0, 0, 0], 
                      [0, 0, -1]], dtype=np.float64)

# animal.png de hata veriyor
# libpng warning: iCCP: known incorrect sRGB profile

def applyKernel(img, kernel, n):
    for i in range(n):
        img = cv2.filter2D(img, -1, kernel)
    return img

image = cv2.imread('images/animal.png')
result = applyKernel(image, deriv_rd, 1)
result2 = applyKernel(image, deriv_ld, 1)
result3 = applyKernel(image, deriv_x, 1)
result4 = applyKernel(image, deriv_y, 1)
result5 = result+result2+result3+result4
result6 = applyKernel(result3, gauss, 1)

cv2.imshow('image', image)
cv2.imshow('result-1', result)
cv2.imshow('result-2', result2)
cv2.imshow('result-3', result3)
cv2.imshow('result-4', result4)
cv2.imshow('result-5', result5)
cv2.imshow('result-6', result6)

cv2.waitKey(50000)

