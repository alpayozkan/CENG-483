
import os
import cv2
import numpy as np


# assumes bow.py and the2_data is in the same directory
train_path = 'the2_data/train'
valid_path = 'the2_data/validation'

classes = sorted(os.listdir(train_path))

class_ids = dict({s: c for c,s in enumerate(classes)})
# class_ids["NOT_FOUND"] = -1


sift = cv2.xfeatures2d.SIFT_create(20)

desc_imgs = []

undef = 0

for c in class_ids:
    class_path = train_path + '/' + c + '/'
    imgs = sorted(os.listdir(class_path))
    for img_name in imgs:
        # print(img_name)
        img_name = class_path + img_name
        # print(img_name)
        img = cv2.imread(img_name)
        kpts, des = sift.detectAndCompute(img, None)
        
        if not isinstance(des, np.ndarray): # NOT_FOUND class, since no descriptor available
            undef +=1
            empty_des = np.zeros((1, sift.descriptorSize()), np.float32)
            desc_imgs.append((-1, img_name, empty_des))
        else:
            desc_imgs.append((class_ids[c], img_name, des))

        # print(img.shape)
        # print(des)
        # print(des.shape)
        # print(len(kpts))
        # print(img.shape)
        # print(img_name)

# stack sift-vectors
desc_stack = np.vstack([dsc[2] for dsc in desc_imgs])

# k-means cluster desc_stack => sift vector vocab/dictionary
from scipy.cluster.vq import kmeans
k = 30
iters = 20
codebook, dist = kmeans(desc_stack, k, iters)

# create bow representation
N = len(desc_imgs) # 6000, number of imgs in train set

bow_repr = np.zeros((N, k), dtype=np.float32)
for desc in desc_imgs:
    bow_words, bow_dist = # nearest neighbor implementaiton

