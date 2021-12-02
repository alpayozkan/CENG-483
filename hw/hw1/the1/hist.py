import numpy as np
from PIL import Image

delta = 1e-6

def normalize_hist(hist):
    return (hist / np.sum(hist))

def KL_divg(Q, S): # expects normalized distributions
    divg = 0
    Q = np.ndarray.flatten(Q)
    S = np.ndarray.flatten(S)

    for q,s in zip(Q,S):
        divg += q*np.log2((q+delta)/(s+delta))
    return divg

def calc_hist(arr, intv, bins): # img numpy arr, intv: interval size, bins: number of bins, intv*bins=256, return np.arr histogram
    hist = np.zeros((bins, bins, bins))
    for row in arr:
        for pix in row:
            hist[pix[0]//bins][pix[1]//bins][pix[2]//bins] += 1
    return hist



root_dir = '../dataset/'

file = open(root_dir + 'InstanceNames.txt', 'r')
instance_names = file.read().splitlines()

q_directories = ['query_1', 'query_2', 'query_3']
s_directory = 'support_96'


dir = root_dir + q_directories[0] + '/' + instance_names[0]

img = Image.open(dir)
imgArr = np.array(img)

hist = calc_hist(imgArr, 16, 16)

print(hist)
print(np.sum(hist))
print(hist.shape)

hist_norm = normalize_hist(hist)

print(hist_norm)
print(np.sum(hist_norm))
print(hist_norm.shape)


divg = KL_divg(hist, hist.T)
print(divg)