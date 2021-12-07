import numpy as np
from PIL import Image
import copy

delta = 1e-6

def partition_img3d(img, grid): 
    # partition (Mxgrid)x(Mxgrid) img into M x M pieces each (grid x grid), returns list
    h,w,d = img.shape
    P = [img[i:i+grid,j:j+grid] for i in range(0,h,grid) for j in range(0,w,grid)]
    # assert len(P)==(h/grid)*(w/grid)
    return P

def partition_img2d(img, grid, M):  # I couldnt generalize into 3D, but I plan to update it if I can
    # partition (Mxgrid)x(Mxgrid) img into M x M pieces each (grid x grid)
    height, width = img.shape
    P = img.reshape(M, grid, M, grid)
    P = P.swapaxes(1,2)
    P = P.reshape(M*M,grid,grid)
    return P

def normalize_hist(hist):
    s = np.sum(hist)
    return (hist/s)

def normalize_mult_hists(hists): # hists contains many histograms, np.ndarray
    return np.array([normalize_hist(x) for x in hists])

def KL_divg(Q, S): # expects normalized distributions
    Q = np.ndarray.flatten(Q)
    S = np.ndarray.flatten(S)
    return  np.sum(Q*np.log2((Q+delta)/(S+delta))) / len(Q) # avg KL divg per bin

def calc_hist(arr, intv, bins, type): # generic hist function, type='3d' or 'per_channel'
    if type=='3d':
        return calc_hist_3d(arr, intv, bins)
    elif type=='per_channel':
        return calc_hist_per_channel(arr, intv, bins)
    else:
        raise TypeError("Invalid Histogram Type")

def calc_hist_grid_3d(arr, intv, bins, grid): # returns list of histograms
    P = partition_img3d(arr, grid) # list of partitions
    H = np.array([calc_hist_3d(x, intv, bins) for x in P])
    return H

def calc_hist_grid_per_channel(arr, intv, bins, grid):
    P = partition_img3d(arr, grid) # list of partitions
    H = np.array([calc_hist_per_channel(x, intv, bins) for x in P])
    return H

def calc_hist_3d(arr, intv, bins): # img numpy arr, intv: interval size, bins: number of bins, intv*bins=256, return np.arr histogram
    hist = np.zeros((bins, bins, bins))
    for row in arr:
        for pix in row:
            hist[pix[0]//intv][pix[1]//intv][pix[2]//intv] += 1
    hist = normalize_hist(hist)
    return hist

def calc_hist_per_channel(arr, intv, bins): # img numpy arr, intv: interval size, bins: number of bins, intv*bins=256, return np.arr histogram
    hist = np.zeros((3,bins))
    for row in arr:
        for pix in row:
            hist[0][pix[0]//intv] += 1
            hist[1][pix[1]//intv] += 1
            hist[2][pix[2]//intv] += 1
    hist = normalize_mult_hists(hist)
    return hist

def top1_acc(Q_Res):
    corr = 0
    for q in Q_Res:
        if q == Q_Res[q][0]:
            corr +=1
    return corr/len(Q_Res)

def calc_results_conf1(QS, S, intvs):
    # store results for each query set
    Q1_Res,Q2_Res,Q3_Res,S_Res = dict(),dict(),dict(),dict()
    QS_Res = [Q1_Res, Q2_Res, Q3_Res, S_Res]

    acc_qnt_qry = [[] for i in range(len(intvs))]

    for inv,acc_query in zip(intvs, acc_qnt_qry):
        # intv x bins = 256
        bins = 256//inv    
        S_hists = dict()

        # Normalize data corresponding to the configuration
        for s in S: # for each img in the support set
            S_hists[s] = calc_hist_3d(S[s], inv, bins)

        for Q,Q_Res in zip(QS,QS_Res): # for each query
            for q in Q: # for each img in the query set
                q_hist = calc_hist_3d(Q[q], inv, bins)

                hist_diff = dict()
                for s in S:
                    hist_diff[s] = KL_divg(q_hist, S_hists[s])
                # get the best matching with lowest kl divg
                argmin_hist = min(hist_diff, key=hist_diff.get)
                min_hist = hist_diff[argmin_hist]
                # store the matching
                Q_Res[q] = (argmin_hist, min_hist)
            # calculate acc for the query-i
            acc = top1_acc(Q_Res)
            acc_query.append(acc)
    return acc_qnt_qry

def calc_results_conf2(QS, S, intvs):
    # store results for each query set
    Q1_Res,Q2_Res,Q3_Res,S_Res = dict(),dict(),dict(),dict()
    QS_Res = [Q1_Res, Q2_Res, Q3_Res, S_Res]

    acc_qnt_qry = [[] for i in range(len(intvs))]

    for inv,acc_query in zip(intvs, acc_qnt_qry):
        # intv x bins = 256
        bins = 256//inv    
        S_hists = dict()
        
        # Normalize data corresponding to the configuration
        for s in S: # for each img in the support set
            S_hists[s] = calc_hist_per_channel(S[s], inv, bins)

        for Q,Q_Res in zip(QS,QS_Res): # for each query
            for q in Q: # for each img in the query set
                q_hist = calc_hist_per_channel(Q[q], inv, bins)

                hist_diff = dict()
                for s in S:
                    hist_diff[s] = KL_divg(q_hist, S_hists[s])
                # get the best matching with lowest kl divg
                argmin_hist = min(hist_diff, key=hist_diff.get)
                min_hist = hist_diff[argmin_hist]
                # store the matching
                Q_Res[q] = (argmin_hist, min_hist)
            # calculate acc for the query-i
            acc = top1_acc(Q_Res)
            acc_query.append(acc)
    return acc_qnt_qry

def calc_results_conf3(QS, S, grids, intv): # grids = [48,24,16,12] grids
    # store results for each query set
    Q1_Res,Q2_Res,Q3_Res,S_Res = dict(),dict(),dict(),dict()
    QS_Res = [Q1_Res, Q2_Res, Q3_Res, S_Res]

    acc_qnt_qry = [[] for i in range(len(grids))]
    
    bins = 256//intv

    for grid,acc_query in zip(grids, acc_qnt_qry):
        S_hists = dict()
        
        # Normalize data corresponding to the configuration
        for s in S: # for each img in the support set
            S_hists[s] = calc_hist_grid_3d(S[s], intv, bins, grid)

        for Q,Q_Res in zip(QS,QS_Res): # for each query
            for q in Q: # for each img in the query set
                q_hist = calc_hist_grid_3d(Q[q], intv, bins, grid)
                
                hist_diff = dict()
                for s in S:
                    hist_diff[s] = KL_divg(q_hist, S_hists[s])
                # get the best matching with lowest kl divg
                argmin_hist = min(hist_diff, key=hist_diff.get)
                min_hist = hist_diff[argmin_hist]
                # store the matching
                Q_Res[q] = (argmin_hist, min_hist)
            # calculate acc for the query-i
            acc = top1_acc(Q_Res)
            acc_query.append(acc)
    return acc_qnt_qry

def calc_results_conf4(QS, S, grids, intv): # grids = [48,24,16,12] grids
    # store results for each query set
    Q1_Res,Q2_Res,Q3_Res,S_Res = dict(),dict(),dict(),dict()
    QS_Res = [Q1_Res, Q2_Res, Q3_Res, S_Res]

    acc_qnt_qry = [[] for i in range(len(grids))]
    
    bins = 256//intv

    for grid,acc_query in zip(grids, acc_qnt_qry):
        S_hists = dict()
        
        # Normalize data corresponding to the configuration
        for s in S: # for each img in the support set
            S_hists[s] = calc_hist_grid_per_channel(S[s], intv, bins, grid)

        for Q,Q_Res in zip(QS,QS_Res): # for each query
            for q in Q: # for each img in the query set
                q_hist = calc_hist_grid_per_channel(Q[q], intv, bins, grid)
                
                hist_diff = dict()
                for s in S:
                    hist_diff[s] = KL_divg(q_hist, S_hists[s])
                # get the best matching with lowest kl divg
                argmin_hist = min(hist_diff, key=hist_diff.get)
                min_hist = hist_diff[argmin_hist]
                # store the matching
                Q_Res[q] = (argmin_hist, min_hist)
            # calculate acc for the query-i
            acc = top1_acc(Q_Res)
            acc_query.append(acc)
    return acc_qnt_qry

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


QS = [Q_1, Q_2, Q_3, copy.deepcopy(S)]

######################################################################################################
# config-1, whole histogram 3D

#intvs = [16, 32, 64, 128]

#config_1_res = calc_results_conf1(QS, S, intvs)

######################################################################################################
# config-2, per channel histogram

intvs = [8, 16, 32, 64, 128]

config_2_res = calc_results_conf2(QS, S, intvs)

######################################################################################################
# Part 3-5



grid_intvs = [48, 24, 16, 12]

# pick best configs for 3d and per channel

intvs_3d = [16, 32, 64, 128] # all
inv_3d = 64 # best

intvs_per_ch = [8, 16, 32, 64, 128] # all
inv_per_ch = 32 # best

# config_3_res = [calc_results_conf3(QS, S, grid_intvs, inv) for inv in intvs_3d]
# config_4_res = [calc_results_conf4(QS, S, grid_intvs, inv) for inv in intvs_per_ch]


# butun interval lara bi bak