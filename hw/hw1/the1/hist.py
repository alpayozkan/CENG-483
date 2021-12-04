from matplotlib.pyplot import hist
import numpy as np
from PIL import Image
from numpy.lib.function_base import _calculate_shapes

delta = 1e-6

def normalize_hist(hist):
    return (hist / np.sum(hist))

def KL_divg(Q, S): # expects normalized distributions
    divg = 0
    Q = np.ndarray.flatten(Q)
    S = np.ndarray.flatten(S)
    return  np.sum(Q*np.log2((Q+delta)/(S+delta)))

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

# dictionaries label: np.arr for each img
# Q: query set, S: support set
Q_1 = dict()
Q_2 = dict()
Q_3 = dict()
S = dict()

# read data into Q, S
for inst in instance_names:
    q1_dir = root_dir + q_directories[0] + '/' + inst
    q1_img = Image.open(q1_dir)
    q1_arr = np.array(q1_img)
    Q_1[inst] = q1_arr

    q2_dir = root_dir + q_directories[1] + '/' + inst
    q2_img = Image.open(q2_dir)
    q2_arr = np.array(q2_img)
    Q_2[inst] = q2_arr

    q3_dir = root_dir + q_directories[2] + '/' + inst
    q3_img = Image.open(q3_dir)
    q3_arr = np.array(q3_img)
    Q_3[inst] = q3_arr

    s_dir = root_dir + s_directory + '/' + inst
    s_img = Image.open(s_dir)
    s_arr = np.array(s_img)
    S[inst] = s_arr

QS = [Q_1, Q_2, Q_3]

# store results for each query set
Q1_Res = dict()
Q2_Res = dict()
Q3_Res = dict()

QS_Res = [Q1_Res, Q2_Res, Q3_Res]

# Start histogram computations for each configuration

# Normalize data corresponding to the configuration, generically
# config-1, whole histogram

res = dict()
intvs = [16, 32, 64, 128]

S_hists = dict()
for s in S: # for each img in the support set
    S_hists[s] = normalize_hist(calc_hist(S[s], 16, 16))

for Q,Q_Res in zip(QS,QS_Res): # for each query
    for q in Q: # for each img in the query set
        q_hist = normalize_hist(calc_hist(Q[q], 16, 16))
        hist_diff = dict()
        for s in S:
            hist_diff[s] = KL_divg(q_hist, S_hists[s])
        argmin_hist = min(hist_diff, key=hist_diff.get)
        min_hist = hist_diff[argmin_hist]
        Q_Res[q] = (argmin_hist, min_hist)
            
print(Q1_Res)
print(Q2_Res)
print(Q3_Res)            






"""
debug
"""
"""
for q_folder in q_directories:
    for q_inst in instance_names:
        q_dir = root_dir + q_folder + '/' + q_inst
        q_img = Image.open(q_dir)
        q_arr = np.array(q_img)
        q_arr_norm = normalize_hist(q_arr)

        for s_inst in instance_names:
            sdir = root_dir + s_directory + '/' + s_inst
            s_img = Image.open(sdir)
            s_arr = np.array(s_img)
            s_arr_norm = normalize_hist(s_arr)
            divg = KL_divg(q_arr_norm, s_arr_norm)
"""