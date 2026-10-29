#!/usr/bin/env python3
"""
data[t][y][x] ~ eps[y][x] * avg[cy+y-center_y_t][cx+x-center_x_t]
avg profile
eps[y][x] efficiency

sigma_stat[y][x] = sqrt( sum_t data*avg^2 ) / sum_t avg^2
sigma_tot  = scale * sigma_stat,  scale = sqrt(max(chi2/ndf, 1))
sigma_sys = sqrt(sigma_tot^2 - sigma_stat^2)

files:
epsilon.txt epsilon cointed x y eps sigma_stat sigma_sys sigma_tot
avg_data.npy counted profile
"""

import os
import numpy as np
import scipy.stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

FILE_NAME     = "data/data"
CENTER_FILE   = "data/centers.txt"
REAL_EPS_FILE = "epsilon_real.txt"

PADS_X        = 32
PADS_Y        = 20
TIME_SIZE     = (PADS_X + 10 * 2) * (PADS_Y + 5 * 2)
CYCLES        = 50
TOL_EPS       = 1e-6
USE_SMOOTH    = False
SMOOTH_SIGMA  = 1.0

# file reading

# data
data = []
for num in range(TIME_SIZE):
    frame = np.zeros((PADS_Y, PADS_X))
    path = f"{FILE_NAME}{num}.txt"
    if not os.path.exists(path):
        raise FileNotFoundError(f"Не найден файл {path}")
    with open(path, "r") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            x, y, v = int(parts[0]), int(parts[1]), float(parts[2])
            if 0 <= y < PADS_Y and 0 <= x < PADS_X:
                frame[y, x] = v
    data.append(frame)

print(f"time_size = {TIME_SIZE}, len(data) = {len(data)}")
print(f"shape data[0] = {data[0].shape}")
print(f"max data = {np.max(data):.3f}")
print(f"max data[0] = {np.max(data[0]):.3f}, "
      f"max data[-1] = {np.max(data[-1]):.3f}")

# centers
center = []
with open(CENTER_FILE, "r") as f:
    for line in f:
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        cx_t, cy_t = int(parts[0]), int(parts[1])
        center.append((cx_t, cy_t))   # (x, y)

if len(center) != TIME_SIZE:
    print(f"WARNING: len(center) = {len(center)} != TIME_SIZE = {TIME_SIZE}")
print(f"len(center) = {len(center)}, center[0] = {center[0]}")

# starting counting data
cx = PADS_X // 2
cy = PADS_Y // 2

epsilon  = np.ones((PADS_Y, PADS_X))
avg_data = np.zeros((PADS_Y, PADS_X))

# normalize
max_data_all = max(np.max(d) for d in data)
data_maxmean = [np.max(d) for d in data
                if np.max(d) > max_data_all - max_data_all ** 0.5]
a_norm = float(np.mean(data_maxmean)) if data_maxmean else float(max_data_all)
print(f"a_norm = {a_norm:.3f}  (N_full = {len(data_maxmean)} / {TIME_SIZE})")

# functions
def build_avg(eps):
    a = np.zeros((PADS_Y, PADS_X))
    count = np.zeros((PADS_Y, PADS_X), dtype=int)
    for t in range(TIME_SIZE):
        cxt, cyt = center[t]
        for i in range(PADS_Y):
            for j in range(PADS_X):
                iy = cyt + (i - cy)
                jx = cxt + (j - cx)
                if 0 <= iy < PADS_Y and 0 <= jx < PADS_X:
                    a[i, j] += data[t][iy, jx] / eps[iy, jx]
                    count[i, j] += 1
    
    mask = count > 0
    a[mask] /= count[mask]
    b = np.max(a)
    if b > 0:
        a *= a_norm / b
    if USE_SMOOTH:
        from scipy.ndimage import gaussian_filter
        a = gaussian_filter(a, sigma=SMOOTH_SIGMA)
    return a

def count_eps(avg):
    eps = np.ones((PADS_Y, PADS_X))
    for i in range(PADS_Y):
        for j in range(PADS_X):
            s1 = 0.0
            s2 = 0.0
            for t in range(TIME_SIZE):
                cxt, cyt = center[t]
                iy = cy + i - cyt
                jx = cx + j - cxt
                if 0 <= iy < PADS_Y and 0 <= jx < PADS_X and data[t][i, j] > 0.01:
                    s1 += data[t][i, j] * avg[iy, jx]
                    s2 += avg[iy, jx] ** 2
            if s2 > 0:
                eps[i, j] = s1 / s2
    m = np.mean(eps)
    if m > 0:
        eps /= m
    return eps

# iterations
print("\niter | max|d eps| | max avg | max data")
for n in range(CYCLES):
    avg_new = build_avg(epsilon)
    eps_new = count_eps(avg_new)

    d_eps = np.max(np.abs(eps_new - epsilon))
    avg_data = avg_new
    epsilon  = eps_new

    print(f"{n:3d}  |  {d_eps:.2e}  |  {np.max(avg_data):.1f}  |  {np.max(data):.1f}")

    if d_eps < TOL_EPS:
        break
else:
    print("epsilon wasn't found")

# statistic errors
numers = np.zeros((PADS_Y, PADS_X))
denoms = np.zeros((PADS_Y, PADS_X))
for i in range(PADS_Y):
    for j in range(PADS_X):
        for t in range(TIME_SIZE):
            cxt, cyt = center[t]
            iy = cy + i - cyt
            jx = cx + j - cxt
            if 0 <= iy < PADS_Y and 0 <= jx < PADS_X and data[t][i, j] > 0.01:
                numers[i, j] += data[t][i, j] * avg_data[iy, jx] ** 2
                denoms[i, j] += avg_data[iy, jx] ** 2

eps_errors_stat = np.full((PADS_Y, PADS_X), np.inf)
mask_ok = denoms > 0
eps_errors_stat[mask_ok] = np.sqrt(numers[mask_ok]) / denoms[mask_ok]

# modeling
have_real = os.path.exists(REAL_EPS_FILE)
if have_real:
    real_eps = np.zeros((PADS_Y, PADS_X))
    n_bad = 0
    with open(REAL_EPS_FILE, "r") as f:
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            parts = ln.split()
            if len(parts) < 3:
                continue
            
            xx, yy, vv = int(parts[0]), int(parts[1]), float(parts[2])
            if 0 <= yy < PADS_Y and 0 <= xx < PADS_X:
                real_eps[yy, xx] = vv
            else:
                n_bad += 1
    if n_bad:
        print(str(n_bad) + "points out of range")

    diff = epsilon - real_eps
    chi2_stat = np.sum((diff / eps_errors_stat) ** 2)
    n_pix = PADS_X * PADS_Y
    ndf = n_pix - 1
    prob_stat = scipy.stats.chi2.sf(chi2_stat, ndf)

    scale = np.sqrt(max(chi2_stat / ndf, 1.0))
    eps_errors     = eps_errors_stat * scale
    eps_errors_sys = np.sqrt(np.maximum(eps_errors ** 2 - eps_errors_stat ** 2, 0.0))

    chi2_tot = np.sum((diff / eps_errors) ** 2)
    prob_tot = scipy.stats.chi2.sf(chi2_tot, ndf)

    # ndf_eff autocorrelations
    r = (diff - diff.mean()).flatten()
    acf = np.correlate(r, r, mode='full')
    acf /= acf[len(acf) // 2]
    k = 1
    while k < len(acf) // 2 and acf[len(acf) // 2 + k] > 0:
        k += 1
    sum_rho = np.sum(acf[len(acf) // 2 + 1 : len(acf) // 2 + k])
    ndf_eff = n_pix / (1 + 2 * sum_rho) if sum_rho > 0 else n_pix

    # minuit
    bin_nums = np.arange(n_pix)
    def const_model(x, b):
        return b * np.ones_like(x, dtype=float)
    ls = LeastSquares(bin_nums, diff.flatten(), eps_errors.flatten(), const_model)
    m = Minuit(ls, b=0.0)
    m.migrad(); m.hesse()
    b_val, b_err = m.values['b'], m.errors['b']
else:
    eps_errors = eps_errors_stat.copy()
    eps_errors_sys = np.zeros_like(eps_errors_stat)
    chi2_stat = chi2_tot = prob_stat = prob_tot = np.nan
    ndf = ndf_eff = np.nan
    scale = 1.0
    b_val = b_err = np.nan

# ----------------------------- вывод результатов -----------------------------
print("\n    results   ")
print(f"converged: {d_eps < TOL_EPS}, iter = {n}")
print(f"epsilon: min = {np.min(epsilon):.4f}, max = {np.max(epsilon):.4f}, "
      f"mean = {np.mean(epsilon):.4f}")
finite_stat = eps_errors_stat[np.isfinite(eps_errors_stat)]
print(f"median sigma_stat = {np.median(finite_stat):.3e}  "
      f"(min = {np.min(finite_stat):.3e}, max = {np.max(finite_stat):.3e})")

if have_real:
    print(f"diff: mean = {diff.mean():.3e}, std = {diff.std(ddof=1):.3e}")
    print(f"corr(eps, real)          = "
          f"{np.corrcoef(epsilon.flatten(), real_eps.flatten())[0,1]:.5f}")
    print(f"median |diff|/sigma_stat = "
          f"{np.median(np.abs(diff)/eps_errors_stat):.3f}")
    print(f"ndf = {ndf}, ndf_eff = {ndf_eff:.1f}")
    print(f"chi2_stat = {chi2_stat:.2f}, chi2/ndf = {chi2_stat/ndf:.3f}, "
          f"prob = {prob_stat:.3e}")
    print(f"scale (tot/stat) = {scale:.3f}, "
          f"sys/stat = {np.sqrt(max(chi2_stat/ndf - 1.0, 0.0)):.3f}")
    print(f"chi2_tot  = {chi2_tot:.2f}, chi2/ndf = {chi2_tot/ndf:.3f}, "
          f"prob = {prob_tot:.3e}")
    print(f"b = {b_val:.3e} +/- {b_err:.3e}  ->  "
          f"|b|/sigma_b = {abs(b_val)/b_err:.3f}")
else:
    print("(real_eps not found)")

# ----------------------------- запись -----------------------------
with open("epsilon.txt", "w") as f:
    for i in range(PADS_Y):
        for j in range(PADS_X):
            f.write(f"{j} {i} {epsilon[i,j]:.10e} {eps_errors_stat[i,j]:.6e} "
                    f"{eps_errors_sys[i,j]:.6e} {eps_errors[i,j]:.6e}\n")

np.save("avg_data.npy", avg_data)
