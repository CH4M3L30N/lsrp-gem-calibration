#!/usr/bin/env python3
"""
get_epsilon.py — быстрый и точный восстановитель eps.

Модель:
    data[t,i,j] ≈ sat( eps[i,j] · A_t · V(dx, dy), N_sat )
    dx = j − c_x[t],  dy = i − c_y[t]

V(dx,dy) — 2D-эмпирика (усреднение по узлам целочисленной сетки + сглаживание),
eps — карта эффективности пикселей, A_t — амплитуда кадра.

Ключевые решения:
  • data нормируется на ток I (I0 = median(I)).
  • Сетка V строится по ЦЕЛЫМ узлам (dx_c, dy_c), поэтому (0,0) всегда
    в узле — центр V не пустой, χ²_w/M ~ 1.5.
  • Closed-form eps и A_t (bincount/spsolve) на каждой из 3 внешних итераций.
  • Тихонов по eps (разреженный лапласиан, spsolve) вместо пост-сглаживания.
  • σ_stat — по K=4 фолдам (std * sqrt(K-1)), устойчивее split-half.
  • Bootstrap с пересчётом V (--n-boot > 0) — честная систематика, дорого.
  • Насыщение sat(·, N_sat) опционально (--fit-sat), по умолчанию выкл.
  • sigma_1d вычитает пьедестал (медиана краёв профиля).
  • Края (r > r_max или < min_valid кадров) → nan в eps и σ.

Выход: epsilon.txt, avg_data.txt/.npy, V_dxdy.npy/.txt, V_r.txt,
       P_t.npy, eps.npy, A.npy, eps_folds.npy, N_sat.npy, fit_log.txt
"""

import os
import argparse
import time

import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter


# ============================================================
#  Насыщение
# ============================================================
def saturate(true, N_sat):
    if N_sat <= 0:
        return true
    return true / (1.0 + true / N_sat)


def desaturate(obs, N_sat):
    if N_sat <= 0:
        return obs
    return obs / np.maximum(1.0 - obs / N_sat, 1e-6)


# ============================================================
#  Геометрия
# ============================================================
def build_geometry(centers, PADS_Y, PADS_X):
    T = len(centers)
    cxs = np.array([c[0] for c in centers], dtype=float)
    cys = np.array([c[1] for c in centers], dtype=float)
    J = np.arange(PADS_X, dtype=float)[None, None, :]
    I = np.arange(PADS_Y, dtype=float)[None, :, None]
    dx = np.broadcast_to(J - cxs[:, None, None],
                         (T, PADS_Y, PADS_X)).astype(np.float64)
    dy = np.broadcast_to(I - cys[:, None, None],
                         (T, PADS_Y, PADS_X)).astype(np.float64)
    return dx, dy


# ============================================================
#  Эмпирический V(dx,dy) — усреднение по целым узлам + сглаживание
# ============================================================
def empirical_V_2d(dx, dy, data, mask, step=1.0, sigma_smooth=1.0):
    """
    Узлы сетки — целые значения (dx_c, dy_c), (0,0) всегда узел.
    V[iy, ix] = среднее data в узле (dx_c[ix], dy_c[iy]).
    """
    t_idx, i_idx, j_idx = np.nonzero(mask)
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]
    D = data[t_idx, i_idx, j_idx]

    # --- узлы: целые, (0,0) гарантированно в узле ---
    dx_c = np.arange(np.floor(DX.min()), np.ceil(DX.max()) + 1, step)
    dy_c = np.arange(np.floor(DY.min()), np.ceil(DY.max()) + 1, step)

    # ближайший узел (не «в какой бин попал»)
    ix = np.rint((DX - dx_c[0]) / step).astype(int)
    iy = np.rint((DY - dy_c[0]) / step).astype(int)
    ix = np.clip(ix, 0, len(dx_c) - 1)
    iy = np.clip(iy, 0, len(dy_c) - 1)

    H = np.zeros((len(dy_c), len(dx_c)))
    C = np.zeros((len(dy_c), len(dx_c)))
    np.add.at(H, (iy, ix), D)
    np.add.at(C, (iy, ix), 1.0)
    V = np.where(C > 0, H / np.maximum(C, 1e-9), np.nan)

    # заполняем nan ближайшими соседями
    nan_mask = ~np.isfinite(V)
    for _ in range(20):
        if not nan_mask.any():
            break
        V_new = V.copy()
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            V_shift = np.roll(V, (di, dj), axis=(0, 1))
            C_shift = np.roll(C, (di, dj), axis=(0, 1))
            ok = nan_mask & np.isfinite(V_shift) & (C_shift > 0)
            V_new[ok] = V_shift[ok]
        V = V_new
        nan_mask = ~np.isfinite(V)
    V = np.maximum(np.where(np.isfinite(V), V, 0.0), 0.0)

    # сглаживание (sigma в узлах)
    V = gaussian_filter(V, sigma=sigma_smooth)
    Vmax = V.max()
    if Vmax <= 0:
        raise ValueError("V_emp is zero everywhere")
    V = V / Vmax

    return dx_c, dy_c, V, C


# ============================================================
#  Разреженный лапласиан (Neumann)
# ============================================================
def build_laplacian_sparse(Y, X):
    n = Y * X
    idx = np.arange(n).reshape(Y, X)
    rows, cols, vals = [], [], []
    for i in range(Y):
        for j in range(X):
            k = idx[i, j]
            nb = 0
            for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ii, jj = i + di, j + dj
                if 0 <= ii < Y and 0 <= jj < X:
                    rows.append(k); cols.append(idx[ii, jj]); vals.append(-1.0)
                    nb += 1
            rows.append(k); cols.append(k); vals.append(float(nb))
    return coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()


# ============================================================
#  Closed-form eps и A_t
# ============================================================
def solve_eps(D_lin, W_lin, av, lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X):
    rhs = np.bincount(lin_eps, weights=W_lin * D_lin * av, minlength=N_eps)
    diag = np.bincount(lin_eps, weights=W_lin * av * av, minlength=N_eps)
    if lam_eps > 0 and L is not None:
        M = diags(diag, 0) + lam_eps * diag.mean() * (L.T @ L)
        eps = spsolve(M.tocsc(), rhs)
    else:
        eps = np.where(diag > 0, rhs / np.maximum(diag, 1e-300), 1.0)
    eps = np.maximum(eps, 1e-6).reshape(PADS_Y, PADS_X)
    return eps.ravel()


def solve_A(D_lin, W_lin, ev, lin_A, T):
    num = np.bincount(lin_A, weights=W_lin * D_lin * ev, minlength=T)
    den = np.bincount(lin_A, weights=W_lin * ev * ev, minlength=T)
    return np.maximum(
        np.where(den > 0, num / np.maximum(den, 1e-300), 1e-6), 1e-6)


# ============================================================
#  Один фит eps/A на (под)выборке кадров
# ============================================================
def _fit_eps_A(D, Wf, V_tab, lin_eps, lin_A, T, N_eps,
               L, lam_eps, PADS_Y, PADS_X,
               n_inner=3, fit_sat=False, n_sat_init=0.0):
    n_sat = float(n_sat_init) if fit_sat else 0.0
    eps = np.ones(N_eps)

    def loss_N(N_sat):
        A_ = solve_A(desaturate(D, N_sat), Wf,
                     eps[lin_eps] * V_tab, lin_A, T)
        model = saturate(eps[lin_eps] * A_[lin_A] * V_tab, N_sat)
        return float(np.sum(Wf * (D - model) ** 2))

    for _ in range(n_inner):
        A = solve_A(desaturate(D, n_sat), Wf,
                    eps[lin_eps] * V_tab, lin_A, T)
        eps = solve_eps(desaturate(D, n_sat), Wf, A[lin_A] * V_tab,
                        lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X)
        m = eps.mean()
        eps /= max(m, 1e-12)
        A *= m
        if fit_sat:
            res = minimize_scalar(loss_N, bounds=(1.0, 1e9), method="bounded",
                                  options={"xatol": 1.0})
            n_sat = float(res.x)

    A = solve_A(desaturate(D, n_sat), Wf,
                eps[lin_eps] * V_tab, lin_A, T)
    return eps.reshape(PADS_Y, PADS_X), A, n_sat


# ============================================================
#  Основной фит
# ============================================================
def fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
            n_iter=3, lam_eps=0.01, step=1.0, sigma_smooth=1.0,
            fit_sat=False, n_sat_init=0.0, log=print):
    T = data.shape[0]
    N_eps = PADS_Y * PADS_X
    W = 1.0 / np.maximum(data_raw, 1.0)

    t_idx, i_idx, j_idx = np.nonzero(mask)
    D = data[t_idx, i_idx, j_idx]
    Wf = W[t_idx, i_idx, j_idx]
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]
    lin_eps = i_idx * PADS_X + j_idx
    lin_A = t_idx
    M = D.size
    log(f"fit points: M = {M} / {T * PADS_Y * PADS_X} "
        f"({100.0 * M / (T * PADS_Y * PADS_X):.1f}%)")

    # ---------- V(dx,dy) ----------
    dx_c, dy_c, V_grid, C_grid = empirical_V_2d(
        dx, dy, data, mask, step=step, sigma_smooth=sigma_smooth)
    frac_filled = (C_grid > 0).sum() / C_grid.size
    log(f"V_emp 2D: shape={V_grid.shape}, "
        f"bins with data: {(C_grid > 0).sum()} / {C_grid.size} "
        f"({100.0 * frac_filled:.1f}%)")
    # центр гарантированно в узле
    ix0 = int(np.argmin(np.abs(dx_c - 0.0)))
    iy0 = int(np.argmin(np.abs(dy_c - 0.0)))
    log(f"V_emp(0,0) = {V_grid[iy0, ix0]:.4f}  "
        f"(dx_c[{ix0}]={dx_c[ix0]:.2f}, dy_c[{iy0}]={dy_c[iy0]:.2f})")

    interp = RegularGridInterpolator(
        (dy_c, dx_c), V_grid, bounds_error=False, fill_value=0.0)
    V_tab = interp(np.column_stack([DY, DX]))
    log(f"V_tab: max={V_tab.max():.3f}, mean={V_tab.mean():.3f}")

    # ---------- инициализация ----------
    L = build_laplacian_sparse(PADS_Y, PADS_X) if lam_eps > 0 else None
    n_sat = float(n_sat_init) if fit_sat else 0.0
    eps = np.ones(N_eps)
    A = solve_A(D, Wf, eps[lin_eps] * V_tab, lin_A, T)

    def loss_N(N_sat):
        model = saturate(eps[lin_eps] * A[lin_A] * V_tab, N_sat)
        return float(np.sum(Wf * (D - model) ** 2))

    # ---------- цикл ----------
    prev = None
    t0 = time.time()
    for it in range(n_iter):
        eps = solve_eps(desaturate(D, n_sat), Wf, A[lin_A] * V_tab,
                        lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X)
        m = eps.mean()
        eps /= max(m, 1e-12)
        A *= m
        A = solve_A(desaturate(D, n_sat), Wf,
                    eps[lin_eps] * V_tab, lin_A, T)
        if fit_sat:
            res = minimize_scalar(loss_N, bounds=(1.0, 1e9), method="bounded",
                                  options={"xatol": 1.0})
            n_sat = float(res.x)
        model = saturate(eps[lin_eps] * A[lin_A] * V_tab, n_sat)
        loss_v = float(np.sum(Wf * (D - model) ** 2))
        log(f"  it {it:2d}: loss={loss_v:.4e}"
            + (f" N_sat={n_sat:.1f}" if fit_sat else "")
            + f" time={time.time() - t0:.2f}s")
        if prev is not None and abs(loss_v - prev) / max(abs(prev), 1e-30) < 1e-6:
            log(f"converged at it {it}")
            break
        prev = loss_v
    log(f"fit time: {time.time() - t0:.2f}s")

    # ---------- финал ----------
    eps = solve_eps(desaturate(D, n_sat), Wf, A[lin_A] * V_tab,
                    lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X)
    m = eps.mean()
    eps /= max(m, 1e-12)
    A *= m
    A = solve_A(desaturate(D, n_sat), Wf,
                eps[lin_eps] * V_tab, lin_A, T)
    eps_2d = eps.reshape(PADS_Y, PADS_X)

    # P_t на полной сетке
    P_t = np.zeros((T, PADS_Y, PADS_X))
    for t in range(T):
        pt = A[t] * interp(np.column_stack([dy[t].ravel(), dx[t].ravel()]))
        P_t[t] = pt.reshape(PADS_Y, PADS_X)

    return {"eps": eps_2d, "A": A, "P_t": P_t, "N_sat": n_sat,
            "V_tab": V_tab, "dx_c": dx_c, "dy_c": dy_c, "V_grid": V_grid,
            "dx": dx, "dy": dy, "mask": mask,
            "t_idx": t_idx, "i_idx": i_idx, "j_idx": j_idx,
            "D": D, "W": Wf, "lin_eps": lin_eps, "lin_A": lin_A}


# ============================================================
#  σ_stat по K фолдам
# ============================================================
def kfold_stat(res, PADS_Y, PADS_X, lam_eps, K=4, log=print):
    T = int(res["lin_A"].max()) + 1
    N_eps = PADS_Y * PADS_X
    D = res["D"]; Wf = res["W"]; V_tab = res["V_tab"]
    lin_eps = res["lin_eps"]; lin_A = res["lin_A"]
    n_sat = res["N_sat"]
    D_lin = desaturate(D, n_sat) if n_sat > 0 else D
    L = build_laplacian_sparse(PADS_Y, PADS_X) if lam_eps > 0 else None

    folds = np.zeros((K, PADS_Y, PADS_X))
    for k in range(K):
        sel = (lin_A % K) == k
        eps_k, _, _ = _fit_eps_A(
            D_lin[sel], Wf[sel], V_tab[sel],
            lin_eps[sel], lin_A[sel], T, N_eps,
            L, lam_eps, PADS_Y, PADS_X, n_inner=3,
            fit_sat=False, n_sat_init=n_sat)
        folds[k] = eps_k

    eps_stat = np.std(folds, axis=0, ddof=1) * np.sqrt(K - 1)
    log(f"kfold({K}): median σ_stat = {np.nanmedian(eps_stat):.3e}")
    return eps_stat, folds


# ============================================================
#  Bootstrap с пересчётом V (опционально)
# ============================================================
def bootstrap_full(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
                   lam_eps, step, sigma_smooth, N_BOOT, seed=42, log=print):
    rng = np.random.default_rng(seed)
    T = data.shape[0]
    eps_boot = np.zeros((N_BOOT, PADS_Y, PADS_X))
    for b in range(N_BOOT):
        idx = rng.integers(0, T, T)
        r = fit_m12(data[idx], data_raw[idx], dx[idx], dy[idx], mask[idx],
                    PADS_Y, PADS_X, n_iter=3, lam_eps=lam_eps,
                    step=step, sigma_smooth=sigma_smooth,
                    fit_sat=False, n_sat_init=0.0,
                    log=lambda *_: None)
        eps_boot[b] = r["eps"]
    eps_stat = np.std(eps_boot, axis=0, ddof=1)
    log(f"bootstrap({N_BOOT}): median σ_stat = {np.nanmedian(eps_stat):.3e}")
    return eps_stat, eps_boot


# ============================================================
#  main
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/data20261109_143025.txt")
    ap.add_argument("--real-eps", default="epsilon_real.txt")
    ap.add_argument("--pads-x", type=int, default=32)
    ap.add_argument("--pads-y", type=int, default=20)
    ap.add_argument("--pad-mm", type=float, nargs=2, default=None)
    ap.add_argument("--n-iter", type=int, default=3)
    ap.add_argument("--lam-eps", type=float, default=0.01)
    ap.add_argument("--r-max", type=float, default=15.0)
    ap.add_argument("--min-valid", type=int, default=200)
    ap.add_argument("--r-sigma-max", type=float, default=14.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--sigma-smooth", type=float, default=0.5)
    ap.add_argument("--kfold", type=int, default=4,
                    help="K фолдов для σ_stat (0 — split-half)")
    ap.add_argument("--n-boot", type=int, default=0,
                    help=">0: bootstrap с пересчётом V (дорого)")
    ap.add_argument("--fit-sat", action="store_true")
    ap.add_argument("--n-sat-init", type=float, default=0.0)
    ap.add_argument("--out", default=".")
    args = ap.parse_args()

    PADS_X, PADS_Y = args.pads_x, args.pads_y
    CX, CY = PADS_X // 2, PADS_Y // 2

    log_lines = []

    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    # ---------- чтение ----------
    if not os.path.exists(args.data):
        raise FileNotFoundError(args.data)
    frames, centers, I_vals = [], [], []
    n_skip = 0
    need = 7 + PADS_X * PADS_Y
    with open(args.data) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            if len(p) != need:
                n_skip += 1
                continue
            try:
                I = float(p[4]); c_x = int(p[5]); c_y = int(p[6])
            except ValueError:
                n_skip += 1
                continue
            if I <= 0 or not (0 <= c_x <= PADS_X and 0 <= c_y <= PADS_Y):
                n_skip += 1
                continue
            fr = np.array(p[7:], dtype=float)
            if fr.size != PADS_X * PADS_Y or not np.all(np.isfinite(fr)):
                n_skip += 1
                continue
            frames.append(fr); I_vals.append(I); centers.append((c_x, c_y))
    if not frames:
        raise ValueError("No valid frames")

    data_raw = np.array(frames).reshape(-1, PADS_Y, PADS_X)
    I_vals = np.array(I_vals)
    T = data_raw.shape[0]
    I0 = float(np.median(I_vals[I_vals > 0]))
    data = data_raw * (I0 / I_vals)[:, None, None]
    log(f"Read {T} frames (skipped {n_skip})")

    cx_arr = np.array([c[0] for c in centers], dtype=float)
    cy_arr = np.array([c[1] for c in centers], dtype=float)
    log(f"c_x: mean={cx_arr.mean():.2f} std={cx_arr.std():.2f} "
        f"min={cx_arr.min():.0f} max={cx_arr.max():.0f}")
    log(f"c_y: mean={cy_arr.mean():.2f} std={cy_arr.std():.2f} "
        f"min={cy_arr.min():.0f} max={cy_arr.max():.0f}")

    # ---------- геометрия + маска ----------
    dx, dy = build_geometry(centers, PADS_Y, PADS_X)
    r = np.sqrt(dx ** 2 + dy ** 2)
    mask = r < args.r_max
    n_valid_2d = mask.sum(axis=0)
    pix_ok = n_valid_2d >= args.min_valid
    mask &= pix_ok[None, :, :]
    log(f"mask: n_valid center = {n_valid_2d[CY, CX]}, "
        f"valid pixels = {pix_ok.sum()} / {PADS_Y * PADS_X}, "
        f"min_valid = {args.min_valid}")

    # ---------- фит ----------
    log(f"\n=== fit M12 (2D V, step={args.step}, lam_eps={args.lam_eps}, "
        f"fit_sat={args.fit_sat}) ===")
    res = fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
                  n_iter=args.n_iter, lam_eps=args.lam_eps,
                  step=args.step, sigma_smooth=args.sigma_smooth,
                  fit_sat=args.fit_sat, n_sat_init=args.n_sat_init, log=log)
    eps = res["eps"]; A = res["A"]; n_sat = res["N_sat"]
    dx_c = res["dx_c"]; dy_c = res["dy_c"]; V_grid = res["V_grid"]

    # ---------- V(r) model vs V_emp ----------
    t_idx = res["t_idx"]; i_idx = res["i_idx"]; j_idx = res["j_idx"]
    D = res["D"]
    R_pt = np.sqrt(dx[t_idx, i_idx, j_idx] ** 2 + dy[t_idx, i_idx, j_idx] ** 2)
    r_bins = np.linspace(0, args.r_max, 31)
    H, _ = np.histogram(R_pt, bins=r_bins, weights=D)
    C, _ = np.histogram(R_pt, bins=r_bins)
    V_emp_r = np.where(C > 0, H / np.maximum(C, 1e-9), 0.0)
    cb = 0.5 * (r_bins[1:] + r_bins[:-1])

    DXg, DYg = np.meshgrid(dx_c, dy_c, indexing="xy")
    Rg = np.sqrt(DXg ** 2 + DYg ** 2)
    V_r_model = np.zeros(len(r_bins) - 1)
    for k in range(len(r_bins) - 1):
        m = (Rg >= r_bins[k]) & (Rg < r_bins[k + 1]) & (V_grid > 0)
        if m.sum() > 0:
            V_r_model[k] = V_grid[m].mean()
    V_r_model /= max(V_r_model.max(), 1e-12)

    good = V_emp_r > 1e-6
    if good.sum() > 3:
        scale = (np.sum(V_emp_r[good] * V_r_model[good]) /
                 max(np.sum(V_r_model[good] ** 2), 1e-30))
    else:
        scale = 1.0

    log("\n=== V(r) model vs V_emp(r) ===")
    log(f"{'r':>8s} {'V_emp':>12s} {'V_model·scale':>14s} {'ratio':>10s}")
    for k in range(0, len(cb), 2):
        vm = V_r_model[k] * scale
        ratio = vm / V_emp_r[k] if V_emp_r[k] > 1e-12 else 0.0
        log(f"{cb[k]:8.2f} {V_emp_r[k]:12.4e} {vm:14.4e} {ratio:10.3f}")

    # ---------- остатки ----------
    V_tab = res["V_tab"]
    Wf = res["W"]
    model = saturate(eps[i_idx, j_idx] * A[t_idx] * V_tab, n_sat)
    resid = D - model
    log(f"\nresidual: rms={np.sqrt(np.mean(resid ** 2)):.4e}, "
        f"max|r|={np.max(np.abs(resid)):.4e}")
    log(f"χ²_w/M = {np.sum(Wf * resid ** 2) / D.size:.3e}")

    # ---------- σ_stat ----------
    if args.n_boot > 0:
        log(f"\n=== σ_stat (bootstrap {args.n_boot}, с пересчётом V) ===")
        eps_stat, folds = bootstrap_full(
            data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
            lam_eps=args.lam_eps, step=args.step,
            sigma_smooth=args.sigma_smooth,
            N_BOOT=args.n_boot, log=log)
    elif args.kfold > 1:
        log(f"\n=== σ_stat (kfold K={args.kfold}) ===")
        eps_stat, folds = kfold_stat(
            res, PADS_Y, PADS_X, lam_eps=args.lam_eps,
            K=args.kfold, log=log)
    else:
        log("\n=== σ_stat (split-half) ===")
        eps_stat, folds = kfold_stat(
            res, PADS_Y, PADS_X, lam_eps=args.lam_eps, K=2, log=log)

    r_2d = np.sqrt(
        (np.arange(PADS_X, dtype=float)[None, :] - CX) ** 2 +
        (np.arange(PADS_Y, dtype=float)[:, None] - CY) ** 2
    )
    eps_stat[r_2d > args.r_sigma_max] = np.nan
    log(f"σ_stat = nan для r > {args.r_sigma_max}: "
        f"{(r_2d > args.r_sigma_max).sum()} пикселей")

    # ---------- сравнение с эталоном ----------
    if os.path.exists(args.real_eps):
        real = np.zeros((PADS_Y, PADS_X))
        with open(args.real_eps) as f:
            for ln in f:
                p = ln.split()
                if len(p) < 3:
                    continue
                xx, yy, vv = int(p[0]), int(p[1]), float(p[2])
                if 0 <= yy < PADS_Y and 0 <= xx < PADS_X:
                    real[yy, xx] = vv
        diff = eps - real
        good = np.isfinite(eps_stat) & (eps_stat > 1e-12)
        chi2 = float(np.sum((diff[good] / eps_stat[good]) ** 2))
        n_good = int(good.sum())
        ndf = max(n_good - 1, 1)
        scale_f = np.sqrt(max(chi2 / ndf, 1.0))
        eps_sys = np.sqrt(np.maximum(
            (eps_stat * scale_f) ** 2 - eps_stat ** 2, 0.0))
        eps_tot = np.sqrt(eps_stat ** 2 + eps_sys ** 2)
        log(f"\n[compare] corr = "
            f"{np.corrcoef(eps[good], real[good])[0,1]:.5f}")
        log(f"[compare] rms(diff) = {np.sqrt(np.mean(diff[good]**2)):.3e}")
        log(f"[compare] χ² = {chi2:.2f}, n_good = {n_good}, "
            f"χ²/n = {chi2/n_good:.3f}, scale = {scale_f:.3f}")
    else:
        log("\n(real_eps не найден)")
        eps_sys = np.full_like(eps_stat, np.nan)
        eps_tot = eps_stat.copy()

    # ---------- запись ----------
    out = args.out
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "epsilon.txt"), "w") as f:
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {eps[i,j]:.10e} {eps_stat[i,j]:.6e} "
                        f"{eps_sys[i,j]:.6e} {eps_tot[i,j]:.6e}\n")

    P_avg = res["P_t"].mean(axis=0)
    with open(os.path.join(out, "avg_data.txt"), "w") as f:
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {P_avg[i,j]:.10e}\n")
    np.save(os.path.join(out, "avg_data.npy"), P_avg)

    np.save(os.path.join(out, "V_dxdy.npy"), V_grid)
    np.savetxt(os.path.join(out, "V_dxdy.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), V_grid.ravel()],
               header="dx dy V")
    np.savetxt(os.path.join(out, "V_r.txt"),
               np.c_[cb, V_emp_r, V_r_model * scale],
               header="r V_emp V_model_scaled")

    np.save(os.path.join(out, "P_t.npy"), res["P_t"])
    np.save(os.path.join(out, "eps.npy"), eps)
    np.save(os.path.join(out, "A.npy"), A)
    np.save(os.path.join(out, "eps_folds.npy"), folds)
    np.save(os.path.join(out, "N_sat.npy"), np.array(n_sat))

    # ---------- σ пучка (с вычетом пьедестала) ----------
    def sigma_1d(profile, coords, n_edge=3):
        profile = np.asarray(profile, dtype=float)
        bg = np.median(np.r_[profile[:n_edge], profile[-n_edge:]])
        w = np.clip(profile - bg, 0, None)
        tot = w.sum()
        if tot <= 0:
            return np.nan, np.nan, bg
        mean = np.sum(coords * w) / tot
        var = np.sum(((coords - mean) ** 2) * w) / tot
        return np.sqrt(max(var, 0.0)), mean, bg

    prof_x = P_avg[CY, :]
    prof_y = P_avg[:, CX]
    xc = np.arange(PADS_X, dtype=float)
    yc = np.arange(PADS_Y, dtype=float)

    sx_m, mx, bgx = sigma_1d(prof_x, xc)
    sy_m, my, bgy = sigma_1d(prof_y, yc)
    log(f"\n=== beam σ (по P_avg, с вычетом пьедестала) ===")
    log(f"x: σ={sx_m:.4f}px  mean={mx:.3f}  bg={bgx:.3e}")
    log(f"y: σ={sy_m:.4f}px  mean={my:.3f}  bg={bgy:.3e}")
    if args.pad_mm:
        log(f"x: σ={sx_m * args.pad_mm[0]:.4f}mm")
        log(f"y: σ={sy_m * args.pad_mm[1]:.4f}mm")
    if args.fit_sat:
        log(f"\nN_sat = {n_sat:.3e}")

    with open(os.path.join(out, "fit_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
