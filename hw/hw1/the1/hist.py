import numpy as np
from PIL import Image
import copy

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
            hist[pix[0]//intv][pix[1]//intv][pix[2]//intv] += 1
    return hist

def calc_hist_per_channel(arr, intv, bins): # img numpy arr, intv: interval size, bins: number of bins, intv*bins=256, return np.arr histogram
    hist_0 = np.zeros(bins)
    hist_1 = np.zeros(bins)
    hist_2 = np.zeros(bins)

    for row in arr:
        for pix in row:
            hist_0[pix[0]//intv] += 1
            hist_1[pix[1]//intv] += 1
            hist_2[pix[2]//intv] += 1
    return (hist_0,hist_1,hist_2)

def top1_acc(Q_Res):
    corr = 0
    for q in Q_Res:
        if q == Q_Res[q][0]:
            corr +=1
    return corr/len(Q_Res)

def calc_results_conf1(QS, S, intvs):
    # store results for each query set
    Q1_Res = dict()
    Q2_Res = dict()
    Q3_Res = dict()
    S_Res = dict()
    QS_Res = [Q1_Res, Q2_Res, Q3_Res, S_Res]

    acc_qnt_qry = [[] for i in range(len(intvs))]

    for inv,acc_query in zip(intvs, acc_qnt_qry):
        # intv x bins = 256
        bins = 256//inv    
        S_hists = dict()

        # Normalize data corresponding to the configuration
        for s in S: # for each img in the support set
            S_hists[s] = normalize_hist(calc_hist(S[s], inv, bins))

        for Q,Q_Res in zip(QS,QS_Res): # for each query
            for q in Q: # for each img in the query set
                q_hist = normalize_hist(calc_hist(Q[q], inv, bins))
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
    Q1_Res = dict()
    Q2_Res = dict()
    Q3_Res = dict()
    S_Res = dict()
    QS_Res = [Q1_Res, Q2_Res, Q3_Res, S_Res]

    acc_qnt_qry = [[] for i in range(len(intvs))]

    for inv,acc_query in zip(intvs, acc_qnt_qry):
        # intv x bins = 256
        bins = 256//inv    
        S_hists = dict()
        
        # Normalize data corresponding to the configuration
        for s in S: # for each img in the support set
            h_0,h_1,h_2 = calc_hist_per_channel(S[s], inv, bins)
            S_hists[s] = (normalize_hist(h_0),normalize_hist(h_1),normalize_hist(h_2))

        for Q,Q_Res in zip(QS,QS_Res): # for each query
            for q in Q: # for each img in the query set
                h_0,h_1,h_2 = calc_hist_per_channel(Q[q], inv, bins)
                q_hist = (normalize_hist(h_0), normalize_hist(h_1), normalize_hist(h_2))
                hist_diff = dict()
                for s in S:
                    kl_0 = KL_divg(q_hist[0], S_hists[s][0])
                    kl_1 = KL_divg(q_hist[1], S_hists[s][1])
                    kl_2 = KL_divg(q_hist[2], S_hists[s][2])
                    hist_diff[s] = (kl_0+kl_1+kl_2)/3
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

# config-1, whole histogram 3D

#intvs = [16, 32, 64, 128]

#config_1_res = calc_results_conf1(QS, S, intvs)


# config-2, per channel histogram

intvs = [8, 16, 32, 64, 128]

config_2_res = calc_results_conf2(QS, S, intvs)

# Part 3-5

def partition_img(img, grid, M): 
    # partition (Mxgrid)x(Mxgrid) img into M x M pieces each (grid x grid)
    height, width, depth = img.shape
    P = img.reshape(grid, M, -1, grid, depth)
    P = P.swapaxes(1,2)
    P = P.reshape(-1,grid,grid,depth)
    return P