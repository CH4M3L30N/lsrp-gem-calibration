#!/usr/bin/env python3
"""
Оценка epsilon_j (эффективность регистрации) по серии кадров
с гладким пучком. Данные -- пуассоновские отсчёты.

Модель:  data[t][j] ~ Poisson( eps_j * avg[idx] ),  idx = cx + j - center[t]
  avg     -- усреднённый профиль (обновляется итерационно)
  eps_j   -- эффективность регистрации в пикселе j

Ошибки:
  статистическая: sigma_stat[j] = sqrt( sum_t data*avg^2 ) / sum_t avg^2
  полная:         sigma_tot[j]  = scale * sigma_stat[j],
                  scale = sqrt( max(chi2_stat/ndf, 1) )
  систематическая: sigma_sys[j] = sqrt(sigma_tot^2 - sigma_stat^2)

Выходные файлы:
  epsilon.txt            -- eps_j через пробел
  epsilon_iteration.txt  -- j, eps_j, sigma_stat, sigma_sys, sigma_tot
  avg_data.npy           -- усреднённый профиль
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
import scipy.stats
from iminuit import Minuit
from iminuit.cost import LeastSquares

# ----------------------------- параметры -----------------------------
FILE_NAME     = "data/data"
CENTER_FILE   = "data/centers.txt"
REAL_EPS_FILE = "epsilon_real.txt"

FULL_TIME     = 53
PADS_X        = 32
CYCLES        = 50
TOL_EPS       = 1e-6
USE_SMOOTH    = False
SMOOTH_SIGMA  = 1.0

# ----------------------------- чтение данных -----------------------------
data = []
for num in range(FULL_TIME):
    with open(f"{FILE_NAME}{num}.txt", "r") as f:
        col = []
        for line in f:
            ln = line.strip()
            if not ln:
                continue
            parts = ln.split()
            col.append(float(parts[1]))
        data.append(col)

center = []
with open(CENTER_FILE, "r") as f:
    for line in f:
        ln = line.strip()
        if ln:
            center.append(int(ln))

N_PIX = len(data[0])
cx    = N_PIX // 2 - 1

# ----------------------------- инициализация -----------------------------
epsilon  = np.ones(N_PIX)
avg_data = np.zeros(PADS_X)

max_data_all = np.max(data)
data_maxmean = [np.max(data[t]) for t in range(FULL_TIME)
                if np.max(data[t]) > max_data_all - max_data_all ** 0.5]
a_norm = float(np.mean(data_maxmean))

# ----------------------------- вспомогательные функции -----------------------------
def build_avg(eps):
    """Усреднённый профиль по всем кадрам. Индексация с защитой границ."""
    a = np.zeros(PADS_X)
    for t in range(FULL_TIME):
        for jj in range(-PADS_X // 2, PADS_X // 2):
            ix = int(center[t] + jj)
            jx = int(cx + jj)
            if 0 <= ix < len(data[t]) and 0 <= jx < PADS_X:
                a[jx] += data[t][ix] / eps[ix]
    a /= FULL_TIME
    b = np.max(a)
    if b > 0:
        a *= a_norm / b
    if USE_SMOOTH:
        a = gaussian_filter1d(a, sigma=SMOOTH_SIGMA)
    return a

def count_eps(avg):
    """Оценка eps_j по фиксированному avg (МНК для модели data = eps*avg)."""
    eps = np.ones(N_PIX)
    for j in range(N_PIX):
        s1 = 0.0
        s2 = 0.0
        for t in range(FULL_TIME):
            ix = int(cx + j - center[t])
            if 0 <= ix < PADS_X and data[t][j] > 0.01:
                s1 += data[t][j] * avg[ix]
                s2 += avg[ix] ** 2
        if s2 > 0:
            eps[j] = s1 / s2
    m = np.mean(eps)
    if m > 0:
        eps /= m
    return eps

# ----------------------------- итерационный цикл -----------------------------
print("iter | max|d eps| | max avg | max data")
for n in range(CYCLES):
    # разрыв петли: avg по старому eps, eps по новому avg
    avg_new = build_avg(epsilon)
    eps_new = count_eps(avg_new)

    d_eps = np.max(np.abs(eps_new - epsilon))
    avg_data = avg_new
    epsilon  = eps_new

    print(f"{n:3d}  |  {d_eps:.2e}  |  {np.max(avg_data):.1f}  |  {np.max(data):.1f}")

    if d_eps < TOL_EPS:
        print(f"converged at iter {n}, max|d eps| = {d_eps:.2e}")
        break
else:
    print(f"NOT converged in {CYCLES} iters, max|d eps| = {d_eps:.2e}")

# ----------------------------- статистические (пуассоновские) ошибки -----------------------------
numers = np.zeros(N_PIX)
denoms = np.zeros(N_PIX)
for j in range(N_PIX):
    for t in range(FULL_TIME):
        ix = int(cx + j - center[t])
        if 0 <= ix < PADS_X and data[t][j] > 0.01:
            numers[j] += data[t][j] * avg_data[ix] ** 2
            denoms[j] += avg_data[ix] ** 2

eps_errors_stat = np.where(denoms > 0, np.sqrt(numers) / denoms, np.inf)

# ----------------------------- сравнение с эталоном -----------------------------
real_eps = []
with open(REAL_EPS_FILE, "r") as f:
    for line in f:
        ln = line.strip()
        if ln:
            real_eps.append(float(ln.split()[1]))
real_eps = np.array(real_eps)

diff = epsilon - real_eps

# --- статистический chi2 ---
chi2_stat = np.sum((diff / eps_errors_stat) ** 2)
ndf       = N_PIX - 1
prob_stat = scipy.stats.chi2.sf(chi2_stat, ndf)

# --- полная ошибка: scale * sigma_stat ---
# scale = sqrt(chi2/ndf) -- эмпирическая калибровка, отражающая систематику
scale          = np.sqrt(max(chi2_stat / ndf, 1.0))
eps_errors     = eps_errors_stat * scale
eps_errors_sys = np.sqrt(np.maximum(eps_errors ** 2 - eps_errors_stat ** 2, 0.0))

chi2_tot = np.sum((diff / eps_errors) ** 2)
prob_tot = scipy.stats.chi2.sf(chi2_tot, ndf)

# --- ndf_eff по автокорреляции остатка ---
r = diff - diff.mean()
acf = np.correlate(r, r, mode='full')
acf /= acf[len(acf) // 2]
k = 1
while k < len(acf) // 2 and acf[len(acf) // 2 + k] > 0:
    k += 1
sum_rho = np.sum(acf[len(acf) // 2 + 1 : len(acf) // 2 + k])
ndf_eff = N_PIX / (1 + 2 * sum_rho) if sum_rho > 0 else N_PIX

# --- проверка среднего смещения ---
bin_nums = np.arange(N_PIX)
def const_model(x, b):
    return b * np.ones_like(x, dtype=float)

ls = LeastSquares(bin_nums, diff, eps_errors, const_model)
m  = Minuit(ls, b=0.0)
m.migrad()
m.hesse()
b_val, b_err = m.values['b'], m.errors['b']

# ----------------------------- вывод результатов -----------------------------
print("\n=== результаты ===")
print(f"converged: {d_eps < TOL_EPS}, iter = {n}")
print(f"diff: mean = {diff.mean():.3e}, std = {diff.std(ddof=1):.3e}")
print(f"corr(eps, real)         = {np.corrcoef(epsilon, real_eps)[0,1]:.5f}")
print(f"std(eps)                = {np.std(epsilon,  ddof=1):.4f}")
print(f"std(real)               = {np.std(real_eps, ddof=1):.4f}")
print(f"median |diff|/sigma_stat = {np.median(np.abs(diff) / eps_errors_stat):.3f}")
print(f"ndf = {ndf}, ndf_eff = {ndf_eff:.1f}  (correlations: {'none' if ndf_eff > 0.9*ndf else 'present'})")
print()
print(f"chi2_stat = {chi2_stat:.2f}, chi2/ndf = {chi2_stat/ndf:.3f}, "
      f"chi2/ndf_eff = {chi2_stat/ndf_eff:.3f}, prob = {prob_stat:.3e}")
print(f"scale (tot/stat)        = {scale:.3f}")
print(f"sys/stat ratio          = {np.sqrt(max(chi2_stat/ndf - 1.0, 0.0)):.3f}")
print(f"chi2_tot  = {chi2_tot:.2f}, chi2/ndf = {chi2_tot/ndf:.3f}, prob = {prob_tot:.3e}")
print()
print(f"b = {b_val:.3e} +/- {b_err:.3e}  ->  |b|/sigma_b = {abs(b_val)/b_err:.3f}")

# ----------------------------- запись -----------------------------
with open("epsilon.txt", "w") as f:
    f.write(" ".join(str(e) for e in epsilon) + "\n")

with open("epsilon_iteration.txt", "w") as f:
    for j in range(N_PIX):
        f.write(f"{j} {epsilon[j]:.10e} {eps_errors_stat[j]:.6e} "
                f"{eps_errors_sys[j]:.6e} {eps_errors[j]:.6e}\n")

np.save("avg_data.npy", avg_data)
