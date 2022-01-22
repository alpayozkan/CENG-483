from skimage import io, color
from skimage.transform import rescale
import numpy as np

# burdaki functio

def read_image(filename):
    img = io.imread(filename)
    if len(img.shape) == 2:
        img = np.stack([img, img, img], 2)
    return img

# rgb => lab repr
def cvt2Lab(image):
    Lab = color.rgb2lab(image)
    return Lab[:, :, 0], Lab[:, :, 1:]  # L, ab

# lab repr => rgb
def cvt2rgb(image):
    return color.lab2rgb(image)

# bunun amaci ne tam anlayamadim, neden
def upsample(image):
    return rescale(image, 4, mode='constant', order=3)