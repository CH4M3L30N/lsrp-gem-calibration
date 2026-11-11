#!/usr/bin/env python3
"""
data[t][y][x] ~ eps[y][x] * avg[cy + y - center_y_t][cx + x - center_x_t]
avg        -- усреднённый профиль
eps[y][x]  -- эффективность регистрации

sigma_stat[y][x] = sqrt( sum_t data*avg^2 ) / sum_t avg^2
sigma_tot = scale * sigma_stat,  scale = sqrt(max(chi2/ndf, 1))
sigma_sys = sqrt(sigma_tot^2 - sigma_stat^2)

files:
  epsilon.txt  -- x y eps sigma_stat sigma_sys sigma_tot
  avg_data.txt -- x y avg
  avg_data.npy -- профиль (numpy)
"""

import os
import numpy as np
import scipy.stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

# ----------------------------- параметры -----------------------------
FILE_NAME     = "data/data"
CENTER_FILE   = "data/centers.txt"
REAL_EPS_FILE = "epsilon_real.txt"

PADS_X        = 32
PADS_Y        = 20
TIME_SIZE     = 693          # data0.txt .. data692.txt
CYCLES        = 50
TOL_EPS       = 1e-6
USE_SMOOTH    = False
SMOOTH_SIGMA  = 1.0

# ----------------------------- чтение данных -----------------------------
data = []
centers = []
timestamps = []

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

        data_flat = np.array([int(v) for v in parts[7:7+PADS_X*PADS_Y]], dtype=float)
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

# ----------------------------- чтение центров -----------------------------
# centers.txt: <x> <y>   (x = 0..32, y = 0..20)
# ВАЖНО: y в centers.txt отсчитывается СНИЗУ ВВЕРХ,
#        а data[t][y] -- СВЕРХУ ВНИЗ. Переворачиваем y.
center = []
with open(CENTER_FILE, "r") as f:
    for line in f:
        ln = line.strip()
        if not ln:
            continue
        parts = ln.split()
        if len(parts) < 2:
            continue
        xs, ys_raw = int(parts[0]), int(parts[1])
        cxt = xs                          # X не перевёрнут
        cyt = PADS_Y - 1 - ys_raw         # Y перевёрнут: 19 - y_raw
        center.append((cxt, cyt))

if len(center) != TIME_SIZE:
    print(f"WARNING: len(center) = {len(center)} != TIME_SIZE = {TIME_SIZE}")
print(f"len(center) = {len(center)}, center[0] = {center[0]}, "
      f"center[-1] = {center[-1]}")

# ----------------------------- инициализация -----------------------------
cx = PADS_X // 2          # 16
cy = PADS_Y // 2          # 10

epsilon  = np.ones((PADS_Y, PADS_X))
avg_data = np.zeros((PADS_Y, PADS_X))

# нормировочный масштаб
max_data_all = max(np.max(d) for d in data)
data_maxmean = [np.max(d) for d in data
                if np.max(d) > max_data_all - max_data_all ** 0.5]
a_norm = float(np.mean(data_maxmean)) if data_maxmean else float(max_data_all)
print(f"a_norm = {a_norm:.3f}  (N_full = {len(data_maxmean)} / {TIME_SIZE})")

# ----------------------------- вспомогательные функции -----------------------------
def build_avg(eps):
    """Усреднённый профиль. Нормировка на число валидных кадров per-pixel."""
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
    """Оценка eps[i][j] по фиксированному avg."""
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

# ----------------------------- итерационный цикл -----------------------------
print("\niter | max|d eps| | max avg | max data")
for n in range(CYCLES):
    avg_new = build_avg(epsilon)
    eps_new = count_eps(avg_new)

    d_eps = np.max(np.abs(eps_new - epsilon))
    avg_data = avg_new
    epsilon  = eps_new

    print(f"{n:3d}  |  {d_eps:.2e}  |  {np.max(avg_data):.1f}  |  {np.max(data):.1f}")

    if d_eps < TOL_EPS:
        avg_data = build_avg(epsilon)
        print(f"converged at iter {n}, max|d eps| = {d_eps:.2e}")
        break
else:
    print(f"NOT converged in {CYCLES} iters, max|d eps| = {d_eps:.2e}")

# ----------------------------- статистические ошибки -----------------------------
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

    # ndf_eff по автокорреляции остатка
    r = (diff - diff.mean()).flatten()
    acf = np.correlate(r, r, mode='full')
    acf /= acf[len(acf) // 2]
    k = 1
    while k < len(acf) // 2 and acf[len(acf) // 2 + k] > 0:
        k += 1
    sum_rho = np.sum(acf[len(acf) // 2 + 1 : len(acf) // 2 + k])
    ndf_eff = n_pix / (1 + 2 * sum_rho) if sum_rho > 0 else n_pix

    # среднее смещение через Minuit
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

# ----------------------------- вывод -----------------------------
print("\n=== результаты ===")
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
    print("(real_eps не найден -- сравнение пропущено)")

# ----------------------------- запись -----------------------------
with open("epsilon.txt", "w") as f:
    for i in range(PADS_Y):
        for j in range(PADS_X):
            f.write(f"{j} {i} {epsilon[i,j]:.10e} {eps_errors_stat[i,j]:.6e} "
                    f"{eps_errors_sys[i,j]:.6e} {eps_errors[i,j]:.6e}\n")

with open("avg_data.txt", "w") as f:
    for i in range(PADS_Y):
        for j in range(PADS_X):
            f.write(f"{j} {i} {avg_data[i,j]:.10e}\n")

np.save("avg_data.npy", avg_data)

# ----------------------------- sigma по двум осям -----------------------------
def sigma_1d(profile, coords):
    """
    Стандартное отклонение 1D-профиля:
        sigma^2 = sum( (x - <x>)^2 * w ) / sum( w ),  w = profile
    profile : 1D array неотрицательных значений
    coords  : 1D array координат (той же длины)
    Возвращает (sigma, mean, total) или (nan, nan, nan), если sum(profile) <= 0.
    """
    profile = np.asarray(profile, dtype=float)
    coords  = np.asarray(coords,  dtype=float)

    if profile.size < 2:
        return (np.nan, np.nan, np.nan)

    # обрезаем отрицательные значения (на всякий случай)
    w = np.clip(profile, 0.0, None)
    total = w.sum()
    if total <= 0:
        return (np.nan, np.nan, np.nan)

    mean = np.sum(coords * w) / total
    var  = np.sum(((coords - mean) ** 2) * w) / total
    sigma = np.sqrt(max(var, 0.0))
    return (sigma, mean, total)


# ось x: профиль вдоль строки y = cy
x_coords = np.arange(PADS_X, dtype=float)
profile_x = avg_data[cy, :]
sigma_x, mean_x, total_x = sigma_1d(profile_x, x_coords)

# ось y: профиль вдоль столбца x = cx
y_coords = np.arange(PADS_Y, dtype=float)
profile_y = avg_data[:, cx]
sigma_y, mean_y, total_y = sigma_1d(profile_y, y_coords)

# в пикселях профиля (1 пиксель = 1 шаг детектора)
print("\n=== sigma (в шагах детектора / пикселях профиля) ===")
print(f"x-axis (row y={cy}): sigma = {sigma_x:.4f} px, "
      f"mean = {mean_x:.3f}, total = {total_x:.3e}")
print(f"y-axis (col x={cx}): sigma = {sigma_y:.4f} px, "
      f"mean = {mean_y:.3f}, total = {total_y:.3e}")

PAD_MM = [2., 1.]

if PAD_MM is not None:
    print("\n=== sigma (в мм, PAD_MM = {} x {} мм) ===".format(*PAD_MM))
    print(f"x-axis: sigma = {sigma_x * PAD_MM[0]:.4f} mm")
    print(f"y-axis: sigma = {sigma_y * PAD_MM[1]:.4f} mm")
