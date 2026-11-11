#!/usr/bin/env python3
"""
Итоговый get_epsilon.py с Тихоновской регуляризацией.

Читает один файл DATA_FILE формата:
  <ts_start_step> <ts_end_step> <ts_start_data> <ts_end_data> <I> <c_x> <c_y> <data>
  ts_* : "YYYY-MM-DDTHH:MM:SS.mmm" (ISO 8601, без пробела)
  data: 640 int (построчно: y=0: x=0..31, ...)
  c_x, c_y: центр детектора = (16, 10); в центре зелёной зоны c_x=16, c_y=10

Модель:
  data[t][i][j] ≈ eps[i][j] * P_t[iy, jx]
  (iy, jx) = (CY + i - c_y_t, CX + j - c_x_t)
  CY = 10, CX = 16 (центр зелёной зоны)

F = Σ_t Σ_{i,j} (data - eps*P_t)² + λ_P*||∇P_t||² + λ_eps*||∇eps||²

files:
  epsilon.txt   -- x y eps sigma_stat sigma_sys sigma_tot
  avg_data.txt  -- x y P_avg_final
  avg_data.npy  -- P_avg_final (numpy)
  P_t.npy       -- P_t (numpy, TIME_SIZE x PADS_Y x PADS_X)
  eps.npy       -- eps (numpy, PADS_Y x PADS_X)
"""

import os
import numpy as np
import scipy.stats
from scipy.ndimage import gaussian_filter, laplace

# ----------------------------- параметры -----------------------------
DATA_FILE         = "data/data20261109_143025.txt"   # <-- имя файла
REAL_EPS_FILE     = "epsilon_real.txt"

PADS_X            = 32
PADS_Y            = 20
CX                = PADS_X // 2   # 16
CY                = PADS_Y // 2   # 10

LAMBDA_P          = 0.01
LAMBDA_EPS        = 0.01
ETA_EPS           = 0.02
ETA_P             = 0.01
N_ITER            = 500
TOL_EPS           = 1e-4

# ----------------------------- чтение данных -----------------------------
data = []
centers = []
timestamps = []
I_0 = 0.

if not os.path.exists(DATA_FILE):
    raise FileNotFoundError(DATA_FILE)

with open(DATA_FILE, "r") as f:
    for line in f:
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 7 + PADS_X * PADS_Y:
            print(f"WARNING: line too short ({len(parts)} fields)")
            continue

        ts_start_step = parts[0]
        ts_end_step   = parts[1]
        ts_start_data = parts[2]
        ts_end_data   = parts[3]
        I             = float(parts[4])
        c_x           = int(parts[5])
        c_y           = int(parts[6])
        
        if I_0 < 0.000001:
            I_0 = I

        data_flat = np.array([(int(v) / I * I_0) for v in parts[7:7+PADS_X*PADS_Y]], dtype=float)
        frame = data_flat.reshape((PADS_Y, PADS_X))

        data.append(frame)
        centers.append((c_x, c_y))
        timestamps.append((ts_start_step, ts_end_step,
                           ts_start_data, ts_end_data))

TIME_SIZE = len(data)
if TIME_SIZE == 0:
    raise ValueError(f"No valid frames in {DATA_FILE}")

print(f"Read {TIME_SIZE} frames from {DATA_FILE}")
data = np.array(data)   # (TIME_SIZE, PADS_Y, PADS_X)

# ----------------------------- маски и сдвиг -----------------------------
# c_x, c_y уже в системе координат data (y=0 сверху)
mask_t = np.zeros((TIME_SIZE, PADS_Y, PADS_X), dtype=bool)
shift_idx = np.full((TIME_SIZE, PADS_Y, PADS_X, 2), -1, dtype=int)
for t in range(TIME_SIZE):
    cxt, cyt = centers[t]
    for i in range(PADS_Y):
        for j in range(PADS_X):
            iy = CY + i - cyt
            jx = CX + j - cxt
            if 0 <= iy < PADS_Y and 0 <= jx < PADS_X:
                mask_t[t, i, j] = True
                shift_idx[t, i, j] = (iy, jx)

# ----------------------------- инициализация P_avg -----------------------------
P_avg = np.zeros((PADS_Y, PADS_X))
count_P = np.zeros((PADS_Y, PADS_X), dtype=int)
for t in range(TIME_SIZE):
    cxt, cyt = centers[t]
    for di in range(-CY, PADS_Y - CY):
        for dj in range(-CX, PADS_X - CX):
            iy = cyt + di
            jx = cxt + dj
            if 0 <= iy < PADS_Y and 0 <= jx < PADS_X:
                P_avg[CY + di, CX + dj] += data[t][iy, jx]
                count_P[CY + di, CX + dj] += 1
P_avg = np.where(count_P > 0, P_avg / np.maximum(count_P, 1), 0)
P_avg = gaussian_filter(P_avg, sigma=2.0)
b = np.max(P_avg)
if b > 0:
    P_avg *= np.max(data) / b
print(f"P_avg: max = {np.max(P_avg):.3f}")

# ----------------------------- инициализация P_t, eps -----------------------------
P_t = np.array([P_avg.copy() / 2.0 for _ in range(TIME_SIZE)])
eps = np.ones((PADS_Y, PADS_X))

print(f"P_t (init): max = {np.max(P_t):.3f}")

# ----------------------------- градиентный спуск -----------------------------
print("\niter | max|d eps| | F_data | F_reg_P | F_reg_eps | corr")
for n in range(N_ITER):
    # ---------- градиент eps ----------
    grad_eps = np.zeros((PADS_Y, PADS_X))
    for t in range(TIME_SIZE):
        for i in range(PADS_Y):
            for j in range(PADS_X):
                if not mask_t[t, i, j]:
                    continue
                iy, jx = shift_idx[t, i, j]
                p = P_t[t, iy, jx]
                d = data[t, i, j]
                residual = d - eps[i, j] * p
                grad_eps[i, j] += -2.0 * residual * p

    lap_eps = laplace(eps, mode='nearest')
    grad_eps += 2.0 * LAMBDA_EPS * lap_eps

    grad_eps_max = np.max(np.abs(grad_eps))
    if grad_eps_max > 0:
        grad_eps /= grad_eps_max

    eps_new = eps - ETA_EPS * grad_eps
    eps_new = np.clip(eps_new, 0.01, None)
    m = np.mean(eps_new)
    if m > 0:
        eps_new /= m

    # ---------- градиент P_t ----------
    grad_P_t = np.zeros_like(P_t)
    for t in range(TIME_SIZE):
        for i in range(PADS_Y):
            for j in range(PADS_X):
                if not mask_t[t, i, j]:
                    continue
                iy, jx = shift_idx[t, i, j]
                p = P_t[t, iy, jx]
                d = data[t, i, j]
                e = eps[i, j]
                residual = d - e * p
                grad_P_t[t, iy, jx] += -2.0 * residual * e

    for t in range(TIME_SIZE):
        lap_P = laplace(P_t[t], mode='nearest')
        grad_P_t[t] += 2.0 * LAMBDA_P * lap_P

    grad_P_max = np.max(np.abs(grad_P_t))
    if grad_P_max > 0:
        grad_P_t /= grad_P_max

    P_t_new = P_t - ETA_P * grad_P_t
    P_t_new = np.clip(P_t_new, 0.01, None)

    # ---------- мониторинг ----------
    d_eps = np.max(np.abs(eps_new - eps))
    eps = eps_new
    P_t = P_t_new

    F_data = 0.0
    for t in range(TIME_SIZE):
        for i in range(PADS_Y):
            for j in range(PADS_X):
                if not mask_t[t, i, j]:
                    continue
                iy, jx = shift_idx[t, i, j]
                residual = data[t, i, j] - eps[i, j] * P_t[t, iy, jx]
                F_data += residual ** 2

    F_reg_P = 0.0
    for t in range(TIME_SIZE):
        F_reg_P += np.sum(laplace(P_t[t], mode='nearest') ** 2)

    F_reg_eps = np.sum(laplace(eps, mode='nearest') ** 2)

    mean_data = np.mean(data, axis=0)
    corr = np.corrcoef(eps.flatten(), mean_data.flatten())[0,1]

    if n % 10 == 0 or n == N_ITER - 1:
        print(f"{n:3d}  |  {d_eps:.2e}  |  {F_data:.2e}  |  {F_reg_P:.2e}  |  "
              f"{F_reg_eps:.2e}  |  {corr:+.3f}")
    else:
        print(f"{n:3d}  |  {d_eps:.2e}  |  {F_data:.2e}  |  {F_reg_P:.2e}  |  "
              f"{F_reg_eps:.2e}")

    if d_eps < TOL_EPS:
        print(f"converged at iter {n}, max|d eps| = {d_eps:.2e}")
        break
else:
    print(f"NOT converged in {N_ITER} iters, max|d eps| = {d_eps:.2e}")

# ----------------------------- ошибки eps (метод A) -----------------------------
eps_per_frame = np.full((TIME_SIZE, PADS_Y, PADS_X), np.nan)
for t in range(TIME_SIZE):
    for i in range(PADS_Y):
        for j in range(PADS_X):
            if not mask_t[t, i, j]:
                continue
            iy, jx = shift_idx[t, i, j]
            p = P_t[t, iy, jx]
            if p > 0.01 and data[t, i, j] > 0.01:
                eps_per_frame[t, i, j] = data[t, i, j] / p

eps_errors_stat = np.full((PADS_Y, PADS_X), np.inf)
n_valid_eps = np.zeros((PADS_Y, PADS_X), dtype=int)
for i in range(PADS_Y):
    for j in range(PADS_X):
        vals = eps_per_frame[:, i, j]
        vals = vals[np.isfinite(vals)]
        n_valid_eps[i, j] = len(vals)
        if len(vals) > 1:
            eps_errors_stat[i, j] = np.std(vals, ddof=1) / np.sqrt(len(vals))

print(f"\neps: min = {np.min(eps):.4f}, max = {np.max(eps):.4f}, "
      f"mean = {np.mean(eps):.4f}")
finite = eps_errors_stat[np.isfinite(eps_errors_stat)]
print(f"median sigma_stat = {np.median(finite):.3e}  "
      f"(min = {np.min(finite):.3e}, max = {np.max(finite):.3e})")

# ----------------------------- сравнение с эталоном -----------------------------
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
        print(f"WARNING: {n_bad} точек real_eps вне диапазона")

    diff = eps - real_eps
    chi2_stat = np.sum((diff / eps_errors_stat) ** 2)
    n_pix = PADS_X * PADS_Y
    ndf = n_pix - 1
    prob_stat = scipy.stats.chi2.sf(chi2_stat, ndf)

    scale = np.sqrt(max(chi2_stat / ndf, 1.0))
    eps_errors     = eps_errors_stat * scale
    eps_errors_sys = np.sqrt(np.maximum(eps_errors ** 2 - eps_errors_stat ** 2, 0.0))

    chi2_tot = np.sum((diff / eps_errors) ** 2)
    prob_tot = scipy.stats.chi2.sf(chi2_tot, ndf)

    print(f"diff: mean = {diff.mean():.3e}, std = {diff.std(ddof=1):.3e}")
    print(f"corr(eps, real)          = "
          f"{np.corrcoef(eps.flatten(), real_eps.flatten())[0,1]:.5f}")
    print(f"median |diff|/sigma_stat = "
          f"{np.median(np.abs(diff)/eps_errors_stat):.3f}")
    print(f"ndf = {ndf}")
    print(f"chi2_stat = {chi2_stat:.2f}, chi2/ndf = {chi2_stat/ndf:.3f}, "
          f"prob = {prob_stat:.3e}")
    print(f"scale (tot/stat) = {scale:.3f}, "
          f"sys/stat = {np.sqrt(max(chi2_stat/ndf - 1.0, 0.0)):.3f}")
    print(f"chi2_tot  = {chi2_tot:.2f}, chi2/ndf = {chi2_tot/ndf:.3f}, "
          f"prob = {prob_tot:.3e}")
else:
    eps_errors = eps_errors_stat.copy()
    eps_errors_sys = np.zeros_like(eps_errors_stat)
    print("(real_eps не найден -- сравнение пропущено)")

# ----------------------------- запись -----------------------------
with open("epsilon.txt", "w") as f:
    for i in range(PADS_Y):
        for j in range(PADS_X):
            f.write(f"{j} {i} {eps[i,j]:.10e} {eps_errors_stat[i,j]:.6e} "
                    f"{eps_errors_sys[i,j]:.6e} {eps_errors[i,j]:.6e}\n")

P_avg_final = np.mean(P_t, axis=0)
with open("avg_data.txt", "w") as f:
    for i in range(PADS_Y):
        for j in range(PADS_X):
            f.write(f"{j} {i} {P_avg_final[i,j]:.10e}\n")

np.save("avg_data.npy", P_avg_final)
np.save("P_t.npy", P_t)
np.save("eps.npy", eps)

print(f"\nP_avg_final: max = {np.max(P_avg_final):.3f}")

# ----------------------------- sigma по двум осям -----------------------------
def sigma_1d(profile, coords):
    profile = np.asarray(profile, dtype=float)
    coords  = np.asarray(coords,  dtype=float)
    if profile.size < 2:
        return (np.nan, np.nan, np.nan)
    w = np.clip(profile, 0.0, None)
    total = w.sum()
    if total <= 0:
        return (np.nan, np.nan, np.nan)
    mean = np.sum(coords * w) / total
    var  = np.sum(((coords - mean) ** 2) * w) / total
    sigma = np.sqrt(max(var, 0.0))
    return (sigma, mean, total)

x_coords = np.arange(PADS_X, dtype=float)
profile_x = P_avg_final[CY, :]
sigma_x, mean_x, total_x = sigma_1d(profile_x, x_coords)

y_coords = np.arange(PADS_Y, dtype=float)
profile_y = P_avg_final[:, CX]
sigma_y, mean_y, total_y = sigma_1d(profile_y, y_coords)

print("\n=== sigma (в шагах детектора / пикселях профиля) ===")
print(f"x-axis (row y={CY}): sigma = {sigma_x:.4f} px, "
      f"mean = {mean_x:.3f}, total = {total_x:.3e}")
print(f"y-axis (col x={CX}): sigma = {sigma_y:.4f} px, "
      f"mean = {mean_y:.3f}, total = {total_y:.3e}")

PAD_MM = [2., 1.]
if PAD_MM is not None:
    print("\n=== sigma (в мм, PAD_MM = {} x {} мм) ===".format(*PAD_MM))
    print(f"x-axis: sigma = {sigma_x * PAD_MM[0]:.4f} mm")
    print(f"y-axis: sigma = {sigma_y * PAD_MM[1]:.4f} mm")
