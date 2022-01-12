
import os
import cv2
import numpy as np


# assumes bow.py and the2_data is in the same directory
train_path = 'the2_data/train'
valid_path = 'the2_data/validation'

classes = sorted(os.listdir(train_path))

class_ids = dict({s: c for c,s in enumerate(classes)})
# class_ids["NOT_FOUND"] = -1

def custom_sift(sift, img, is_dense=False, grid_size=1):
    if is_dense:
        kp_grids = [cv2.KeyPoint(x, y, grid_size) for y in range(0, img.shape[0], grid_size) for x in range(0, img.shape[1], grid_size)]
        return sift.compute(img, kp_grids)
    else:
        return sift.detectAndCompute(img, None)

# DEFINE SIFT Params Here


# DENSE
is_dense = True
grid_size = 32

# SIFT
nfeatures = 0
nOctaveLayers = 3
contrastThreshold = 0.04
edgeThreshold = 10
sigma = 1.6

#   sift = cv2.xfeatures2d.SIFT_create(
#      nfeatures=nfeatures,
#      nOctaveLayers=nOctaveLayers,
#      contrastThreshold=contrastThreshold,
#      edgeThreshold=edgeThreshold,
#      sigma=sigma)

sift = cv2.SIFT_create(
    nfeatures=nfeatures,
    nOctaveLayers=nOctaveLayers,
    contrastThreshold=contrastThreshold,
    edgeThreshold=edgeThreshold,
    sigma=sigma)

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
        kpts, des = custom_sift(sift, img, is_dense, grid_size)
        
        if not isinstance(des, np.ndarray): # NOT_FOUND class, since no descriptor available
            undef +=1
            empty_des = np.zeros((1, sift.descriptorSize()), np.float32)
            # alternative approach, NOT_FOUND class [-1]
            desc_imgs.append((class_ids[c], img_name, empty_des))
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
# cluster-center dictionary
from scipy.cluster.vq import kmeans, vq
k = 128 # 3 dk surdu
iters = 15
codebook, dist = kmeans(desc_stack, k, iters)



def knn(db, query, k=1, dmetric_id=0): 
# db is stack of vectors
# query vector, one vector that we are trying to match the closest one
# take k closest vectors and let them vote
# returns k vector indices that are closest in db
# distance metric: dmetric
# db: (M,128)
# q: (K, 128)
    def cos_dmetric(u,v):
        return np.dot(u,v)/(np.linalg.norm(u,2)*np.linalg.norm(v,2))
    def euc_dmetric(u,v):
        return np.linalg.norm(u-v, 2)

    dmetrics = [cos_dmetric, euc_dmetric]
    dmetric = dmetrics[dmetric_id]
    
    if len(query.shape)==1:
    # returns [()]
        dist = []
        for i,v in enumerate(db):
            d = dmetric(v, query)
            dist.append((i,d))
        dist.sort(key=lambda x: x[1], reverse=False)
        dist = dist[:k]
    else:
        dist = []
        for q in query:
            dist_q = []
            for i,v in enumerate(db):
                d = dmetric(v, q)
                dist_q.append((i,d))
            dist_q.sort(key=lambda x: x[1], reverse=False)
            dist.append(dist_q[:k])
    # take first k-closest vects
    return dist


# create bow representation
# BOF extraction
N = len(desc_imgs) # 6000, number of imgs in train set 15x400=6000

bow_repr = np.zeros((N, k), dtype=np.float32)
for i,desc in enumerate(desc_imgs):
    # bow_words, bow_dist = vq(desc[2], codebook) # nearest neighbor implementaiton, own implementation or just 1-nn, en yakin centroid
    bow_words_tups = knn(codebook, desc[2], k=1, dmetric_id=1)
    bow_words = [x[0][0] for x in bow_words_tups]
    for word in bow_words: # for each word of bow repr of img
        bow_repr[i][word] += 1  # count number of words selected from bow centroid vectors
                                # bow_repr[i] => desc_imgs[i] histogram

# normalize bow histogram
# standard normalization: mean, std
#from sklearn.preprocessing import StandardScaler
#bow_repr_normd = StandardScaler().fit(bow_repr).transform(bow_repr)
bow_repr_norms = np.linalg.norm(bow_repr,1,axis=1)
bow_repr_normzd = (bow_repr/bow_repr_norms[:,None]) # (6000,15)
# train her bir img icin bow normalized histogramlari

# BOF database i bu oluyor, train img lardan elde edildi
# img id si belli, karisligindaki bow_repr da belli
# test data yi da BOF repr et, train dekilere karsi knn at

# Final step
# CLASSIFICATION
# test img lari al => sift vector => 
# BOF extract et => k-nn arasindan train BOF database ine gore 
# dominate eden class i sec

desc_imgs_test = []

undef_test = 0

# Extract sift descriptors of test imgs
for c in class_ids:
    class_path = valid_path + '/' + c + '/'
    imgs = sorted(os.listdir(class_path))
    for img_name in imgs:
        # print(img_name)
        img_name = class_path + img_name
        # print(img_name)
        img = cv2.imread(img_name)
        kpts, des = custom_sift(sift, img, is_dense, grid_size)
        
        if not isinstance(des, np.ndarray): # NOT_FOUND class, since no descriptor available
            undef_test +=1
            empty_des = np.zeros((1, sift.descriptorSize()), np.float32)
            # alternative approach, NOT_FOUND class [-1]
            desc_imgs_test.append((class_ids[c], img_name, empty_des))
        else:
            desc_imgs_test.append((class_ids[c], img_name, des))

# Extract bow representations of test imgs
N_test = len(desc_imgs_test) # 1500, number of imgs in test set 15x100
k_nn = 8 # neares neighbor parameter

test_acc = dict({i: [0,0] for i,s in enumerate(classes)}) # class: (correct, total)
test_acc[-1] = [0,0] # NOT-FOUND class

bow_repr_test = np.zeros((N_test, k), dtype=np.float32)
for i,desc in enumerate(desc_imgs_test):
    # bow_words, bow_dist = vq(desc_imgs_test[2], codebook) # nearest neighbor implementaiton, own implementation or just 1-nn, en yakin centroid
    bow_words_tups_test = knn(codebook, desc[2], k=1, dmetric_id=1) # take the closest centroid as word for that sift vector
    bow_words_test = [x[0][0] for x in bow_words_tups_test]
    for word in bow_words_test: # for each word of bow repr of img
        bow_repr_test[i][word] += 1  # count number of words selected from bow centroid vectors
                                # bow_repr[i] => desc_imgs[i] histogram

# normalize test histograms
bow_repr_test_norms = np.linalg.norm(bow_repr_test,1,axis=1)
bow_repr_test_normzd = (bow_repr_test/bow_repr_test_norms[:,None]) # (1500,15)

# compare test set against train set
# approx 1-2 min

"""
# TRAIN ACC.
# takes too much time, deactivate this block unless debugging
# vote wrt closest nns and generate class predictions

train_acc = dict({i: [0,0] for i,s in enumerate(classes)}) # class: (correct, total)
train_acc[-1] = [0,0] # NOT-FOUND class

train_preds = knn(bow_repr_normzd, bow_repr_normzd, k=k_nn, dmetric_id=1) # for each test img get knn hists in bow_repr_normzd db 

for i,desc in enumerate(desc_imgs):
    voted_imgs = [x[0] for x in train_preds[i]]
    voted_classes = [desc_imgs[x][0] for x in voted_imgs]
    prediction = max(set(voted_classes), key=voted_classes.count) # selects the most frequent/repeated class with democracy
    ground_truth = desc_imgs[i][0]
    train_acc[ground_truth][0] += (prediction==ground_truth)
    train_acc[ground_truth][1] += 1
"""

# TEST ACC.
# vote wrt closest nns and generate class predictions
test_preds = knn(bow_repr_normzd, bow_repr_test_normzd, k=k_nn, dmetric_id=1) # for each test img get knn hists in bow_repr_normzd db
for i,desc in enumerate(desc_imgs_test):
    voted_imgs = [x[0] for x in test_preds[i]]
    voted_classes = [desc_imgs[x][0] for x in voted_imgs]
    prediction = max(set(voted_classes), key=voted_classes.count) # selects the most frequent/repeated class with democracy
    ground_truth = desc_imgs_test[i][0]
    test_acc[ground_truth][0] += (prediction==ground_truth)
    test_acc[ground_truth][1] += 1

def avg_acc(class_acc):
    # accuracy excluding undef class
    tot = 0
    corr = 0
    for c in class_acc:
        tot += class_acc[c][1]
        corr += class_acc[c][0]
    return corr/tot

def avg_acc_undef(class_acc):
    # accuracy including undef class as incorrect
    tot = 0
    corr = 0
    for c in class_acc:
        tot += class_acc[c][1]
        corr += class_acc[c][0]
    # add undefs to total
    tot += class_acc[-1][1]
    return corr/tot

test_acc_avg = avg_acc(test_acc)
