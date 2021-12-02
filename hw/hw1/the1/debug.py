


hist = calc_hist(imgArr, 16, 16)

print(hist)
print(np.sum(hist))
print(hist.shape)

hist_norm = normalize_hist(hist)

print(hist_norm)
print(np.sum(hist_norm))
print(hist_norm.shape)


divg = KL_divg(hist, hist)
print(divg)
