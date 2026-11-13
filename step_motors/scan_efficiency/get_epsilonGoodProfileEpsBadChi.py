#!/usr/bin/env python3
"""
get_epsilon.py — восстановление карты эффективности пикселей eps
по кадрам со сканирующим пучком.

Модель:
    data[t,i,j] ≈ sat( eps[i,j] · A_t · V(dx, dy), N_sat ) + b[i,j]
    dx = j − c_x[t],  dy = i − c_y[t]

V(dx,dy) — ПАРАМЕТРИЧЕСКИЙ повёрнутый псевдо-Voigt:
    V(dx,dy) = A · [η·L(u/σx, v/σy; γ) + (1−η)·G(u/σx, v/σy)]
    u =  dx·cosθ + dy·sinθ
    v = −dx·sinθ + dy·cosθ

Параметры V: A, σx, σy, θ, η, γ (6 штук).

Ключевые решения:
  • data нормируется на ток I (I0 = median(I)).
  • V параметризуется (повёрнутый псевдо-Voigt) — устраняет
    неоднозначность eps·V и «яму» в eps.
  • warmup_V: V_params фитируются первые N итераций, потом
    фиксируются (чтобы θ не «прыгал» в артефакт).
  • lambda_theta: слабый штраф на угол поворота (1.0 вместо 10),
    чтобы не мешать реальному повороту пучка.
  • detrend_plane: после solve_eps вычитается линейный тренд
    (плоскость a + b·x + c·y). Реального крупномасштабного
    градиента на панели быть не может — это артефакт eps·V.
  • Closed-form eps, A_t и b на каждой внешней итерации.
  • Тихонов по eps (разреженный лапласиан, spsolve).
  • σ_stat — по K=4 фолдам (std * sqrt(K-1)).
  • Bootstrap с пересчётом V (--n-boot > 0).
  • Насыщение sat(·, N_sat) опционально (--fit-sat).
  • Мёртвые пиксели: eps = 0 через большой диагональный штраф.
  • Робастные веса (IRLS/Huber).
  • Аддитивный фон b[i,j]: closed-form, может быть < 0.
  • Диагностика остатков: доли |r|>3σ/5σ, вклад выбросов в χ²,
    карты resid_mean / resid_abs_mean, топ-10 пикселей.

Выход: epsilon.txt, avg_data.txt/.npy, V_params.txt, V_dxdy.txt,
       V_dxdy_amp.txt, V_r.txt, bkg.npy, bkg.txt, P_t.npy, eps.npy,
       A.npy, eps_folds.npy, N_sat.npy, fit_log.txt, resid_mean.npy,
       resid_abs_mean.npy, resid_counts.npy, resid_map.txt
       (+ маска, если --save-dead)
"""

import os
import argparse
import time

import numpy as np
import scipy.stats
from scipy.optimize import minimize_scalar, least_squares
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
#  Параметрический V: повёрнутый псевдо-Voigt
# ============================================================
def _pseudo_voigt(u, v, eta, gamma):
    g = np.exp(-0.5 * (u * u + v * v))
    l = 1.0 / (1.0 + u * u + v * v + gamma * gamma)
    return eta * l + (1.0 - eta) * g


def V_param(dx, dy, A, sx, sy, theta, eta, gamma):
    ct, st = np.cos(theta), np.sin(theta)
    u = (dx * ct + dy * st) / sx
    v = (-dx * st + dy * ct) / sy
    return A * _pseudo_voigt(u, v, eta, gamma)


def fit_V_params(dx_flat, dy_flat, V_obs, w_obs,
                 x0=None, bounds=None, log=print,
                 lambda_theta=1.0, lambda_gamma=0.0):
    if x0 is None:
        x0 = np.array([V_obs.max(), 3.0, 3.0, 0.0, 0.5, 0.5])
    if bounds is None:
        bounds = (
            [0.0,  0.5, 0.5, -np.pi, 0.0, 0.01],
            [np.inf, 20.0, 20.0, np.pi, 1.0, 10.0]
        )

    def resid(p):
        A, sx, sy, th, eta, gam = p
        V_pred = V_param(dx_flat, dy_flat, A, sx, sy, th, eta, gam)
        r = w_obs * (V_obs - V_pred)
        pen_theta = np.sqrt(lambda_theta) * th
        if lambda_gamma > 0:
            pen_gamma = np.sqrt(lambda_gamma) * (gam - 0.5)
            return np.r_[r, pen_theta, pen_gamma]
        return np.r_[r, pen_theta]

    try:
        res = least_squares(resid, x0, bounds=bounds, method="trf",
                            max_nfev=2000, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        A, sx, sy, th, eta, gam = res.x
        V_pred = V_param(dx_flat, dy_flat, A, sx, sy, th, eta, gam)
        rms_resid = float(np.sqrt(np.mean((w_obs * (V_obs - V_pred)) ** 2)))
        log(f"  V params fit: cost={res.cost:.3e}, rms_resid={rms_resid:.4e}, "
            f"A={A:.3f}, sx={sx:.2f}, sy={sy:.2f}, "
            f"theta={np.degrees(th):.2f}°, eta={eta:.3f}, gamma={gam:.3f}")
        return res.x
    except Exception as e:
        log(f"  V params fit FAILED: {e}")
        return x0


def empirical_V_2d(dx, dy, data, mask, step=1.0, sigma_smooth=1.0):
    t_idx, i_idx, j_idx = np.nonzero(mask)
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]
    D = data[t_idx, i_idx, j_idx]

    dx_c = np.arange(np.floor(DX.min()), np.ceil(DX.max()) + 1, step)
    dy_c = np.arange(np.floor(DY.min()), np.ceil(DY.max()) + 1, step)

    ix = np.rint((DX - dx_c[0]) / step).astype(int)
    iy = np.rint((DY - dy_c[0]) / step).astype(int)
    ix = np.clip(ix, 0, len(dx_c) - 1)
    iy = np.clip(iy, 0, len(dy_c) - 1)

    H = np.zeros((len(dy_c), len(dx_c)))
    C = np.zeros((len(dy_c), len(dx_c)))
    np.add.at(H, (iy, ix), D)
    np.add.at(C, (iy, ix), 1.0)
    V = np.where(C > 0, H / np.maximum(C, 1e-9), np.nan)

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
#  Мёртвые пиксели
# ============================================================
def detect_dead(eps, eps_stat, n_valid_2d, PADS_Y, PADS_X,
                thresh_eps, thresh_stat):
    dead = np.isfinite(eps) & (eps < thresh_eps)
    if eps_stat is not None:
        dead &= np.isfinite(eps_stat) & (eps_stat < thresh_stat)
    if n_valid_2d is not None:
        dead &= np.isfinite(eps)
    return dead


def read_dead_file(path, PADS_Y, PADS_X):
    dead = np.zeros((PADS_Y, PADS_X), dtype=bool)
    if not path or not os.path.exists(path):
        return dead
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            if len(p) < 2:
                continue
            try:
                x, y = int(p[0]), int(p[1])
            except ValueError:
                continue
            flag = int(p[2]) if len(p) >= 3 else 1
            if 0 <= y < PADS_Y and 0 <= x < PADS_X and flag:
                dead[y, x] = True
    return dead


def write_dead_file(path, dead, PADS_Y, PADS_X):
    with open(path, "w") as f:
        f.write("# x y dead_flag (1=dead, 0=alive)\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {1 if dead[i, j] else 0}\n")


# ============================================================
#  Closed-form eps, A_t, b
# ============================================================
def solve_eps(D_lin, W_lin, av, lin_eps, N_eps, L, lam_eps,
              PADS_Y, PADS_X, dead=None):
    rhs = np.bincount(lin_eps, weights=W_lin * D_lin * av, minlength=N_eps)
    diag = np.bincount(lin_eps, weights=W_lin * av * av, minlength=N_eps)

    dead_flat = None
    if dead is not None and dead.any():
        dead_flat = dead.ravel()
        rhs[dead_flat] = 0.0

    alive = ~dead_flat if dead_flat is not None else np.ones(N_eps, dtype=bool)

    if lam_eps > 0 and L is not None:
        diag_mean = diag[alive].mean() if alive.any() else 1.0
        if not np.isfinite(diag_mean) or diag_mean <= 0:
            diag_mean = 1.0
        M = diags(diag, 0) + lam_eps * diag_mean * (L.T @ L)
        if dead_flat is not None and dead_flat.any():
            M = M + diags(np.where(dead_flat, 1e6 * max(diag_mean, 1.0), 0.0), 0)
        eps = spsolve(M.tocsc(), rhs)
        if not np.all(np.isfinite(eps)):
            M = M + diags(np.full(N_eps, 1e-9 * max(diag_mean, 1.0)), 0)
            eps = spsolve(M.tocsc(), rhs)
    else:
        eps = np.where(diag > 0, rhs / np.maximum(diag, 1e-300), 1.0)

    eps = np.maximum(eps, 0.0)
    if dead_flat is not None:
        eps[dead_flat] = 0.0
    return eps.reshape(PADS_Y, PADS_X).ravel()


def solve_A(D_lin, W_lin, ev, lin_A, T):
    num = np.bincount(lin_A, weights=W_lin * D_lin * ev, minlength=T)
    den = np.bincount(lin_A, weights=W_lin * ev * ev, minlength=T)
    return np.maximum(
        np.where(den > 0, num / np.maximum(den, 1e-300), 1e-6), 1e-6)


def solve_b(D_lin, W_lin, model_no_b, lin_eps, N_eps, dead=None):
    r = D_lin - model_no_b
    num = np.bincount(lin_eps, weights=W_lin * r, minlength=N_eps)
    den = np.bincount(lin_eps, weights=W_lin, minlength=N_eps)
    b = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    if dead is not None and dead.any():
        b[dead.ravel()] = 0.0
    return b


# ============================================================
#  Робастные веса (Huber)
# ============================================================
def robust_weights(resid, c=3.0):
    r = np.asarray(resid, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma = 1.4826 * mad
    if not np.isfinite(sigma) or sigma <= 0:
        return np.ones_like(r)
    r_scaled = np.abs(r - med) / sigma
    w = np.where(r_scaled <= c, 1.0, c / np.maximum(r_scaled, 1e-12))
    return w


# ============================================================
#  Detrend: вычитание плоскости из eps
# ============================================================
def detrend_plane(eps_2d, dead=None):
    """
    Вычитает из eps наилучшую плоскость (a + b·x + c·y).
    Мёртвые пиксели (dead=True) не учитываются в фите плоскости.
    Реального крупномасштабного градиента на панели быть не может —
    это артефакт eps·V. Detrend убирает его, сохраняя локальные дефекты.
    """
    Y, X = eps_2d.shape
    yy, xx = np.mgrid[0:Y, 0:X]
    A = np.c_[np.ones(Y * X), xx.ravel(), yy.ravel()]
    y = eps_2d.ravel().copy()
    if dead is not None and dead.any():
        mask_alive = ~dead.ravel()
        if mask_alive.sum() > 3:
            coef, *_ = np.linalg.lstsq(A[mask_alive], y[mask_alive], rcond=None)
        else:
            coef = np.array([1.0, 0.0, 0.0])
    else:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    plane = (A @ coef).reshape(Y, X)
    eps_detrended = eps_2d - plane + plane.mean()
    eps_detrended = np.maximum(eps_detrended, 0.0)
    if dead is not None and dead.any():
        eps_detrended[dead] = 0.0
    return eps_detrended


# ============================================================
#  Один фит eps/A/b на (под)выборке кадров
# ============================================================
def _fit_eps_A(D, Wf, V_tab, lin_eps, lin_A, T, N_eps,
               L, lam_eps, PADS_Y, PADS_X,
               n_inner=3, fit_sat=False, n_sat_init=0.0, dead=None,
               fit_bg=True, b_init=None, detrend=True):
    n_sat = float(n_sat_init) if fit_sat else 0.0
    eps = np.ones(N_eps)
    b = b_init.copy() if b_init is not None else np.zeros(N_eps)
    if dead is not None:
        eps[dead.ravel()] = 0.0
        b[dead.ravel()] = 0.0

    def loss_N(N_sat):
        A_ = solve_A(desaturate(D, N_sat), Wf,
                     eps[lin_eps] * V_tab, lin_A, T)
        model = saturate(eps[lin_eps] * A_[lin_A] * V_tab, N_sat) + b[lin_eps]
        return float(np.sum(Wf * (D - model) ** 2))

    for _ in range(n_inner):
        D_nb = D - b[lin_eps]
        A = solve_A(desaturate(D_nb, n_sat), Wf,
                    eps[lin_eps] * V_tab, lin_A, T)
        eps = solve_eps(desaturate(D_nb, n_sat), Wf, A[lin_A] * V_tab,
                        lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X, dead=dead)
        if detrend:
            eps_2d_tmp = eps.reshape(PADS_Y, PADS_X)
            eps_2d_tmp = detrend_plane(eps_2d_tmp, dead=dead)
            eps = eps_2d_tmp.ravel()
        if dead is not None and (~dead).any():
            m = eps[~dead.ravel()].mean()
        else:
            m = eps.mean()
        m = max(m, 1e-12)
        eps /= m
        A *= m
        if dead is not None:
            eps[dead.ravel()] = 0.0
        if fit_bg:
            model_no_b = saturate(eps[lin_eps] * A[lin_A] * V_tab, n_sat)
            b = solve_b(D, Wf, model_no_b, lin_eps, N_eps, dead=dead)
        if fit_sat:
            res = minimize_scalar(loss_N, bounds=(1.0, 1e9), method="bounded",
                                  options={"xatol": 1.0})
            n_sat = float(res.x)

    D_nb = D - b[lin_eps]
    A = solve_A(desaturate(D_nb, n_sat), Wf,
                eps[lin_eps] * V_tab, lin_A, T)
    return eps.reshape(PADS_Y, PADS_X), A, n_sat, b.reshape(PADS_Y, PADS_X)


# ============================================================
#  Основной фит (параметрический V + IRLS + фон + warmup + detrend)
# ============================================================
def fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
            n_iter=3, lam_eps=0.01, step=1.0, sigma_smooth=1.0,
            fit_sat=False, n_sat_init=0.0, log=print,
            dead_init=None, n_valid_2d=None,
            dead_thresh_eps=0.05, dead_thresh_stat=0.01,
            robust=True, robust_c=3.0, n_irls=2,
            fit_bg=True, param_V=True, refit_V=True,
            warmup_V=2, lambda_theta=1.0, detrend=True):
    T = data.shape[0]
    N_eps = PADS_Y * PADS_X
    W_base = 1.0 / np.maximum(data_raw, 1.0)

    t_idx, i_idx, j_idx = np.nonzero(mask)
    D = data[t_idx, i_idx, j_idx]
    Wf_base = W_base[t_idx, i_idx, j_idx]
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]
    lin_eps = i_idx * PADS_X + j_idx
    lin_A = t_idx
    M = D.size
    log(f"fit points: M = {M} / {T * PADS_Y * PADS_X} "
        f"({100.0 * M / (T * PADS_Y * PADS_X):.1f}%)")

    # ---------- V: начальное приближение ----------
    dx_c, dy_c, V_grid0, C_grid = empirical_V_2d(
        dx, dy, data, mask, step=step, sigma_smooth=sigma_smooth)
    log(f"V_emp 2D (init): shape={V_grid0.shape}, "
        f"bins with data: {(C_grid > 0).sum()} / {C_grid.size}")

    iy_max, ix_max = np.unravel_index(np.argmax(V_grid0), V_grid0.shape)
    shift_x = dx_c[ix_max]
    shift_y = dy_c[iy_max]
    log(f"V center offset: dx={shift_x:.2f}, dy={shift_y:.2f}")

    dx = dx - shift_x
    dy = dy - shift_y
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]

    # ---------- V: параметрический фит ----------
    if param_V:
        m_valid = C_grid > 0
        if m_valid.sum() > 10:
            DXg, DYg = np.meshgrid(dx_c - shift_x, dy_c - shift_y, indexing="xy")
            dx_flat = DXg[m_valid].ravel()
            dy_flat = DYg[m_valid].ravel()
            V_obs = V_grid0[m_valid].ravel()
            w_obs = np.sqrt(np.maximum(C_grid[m_valid].ravel(), 1.0))
            w_obs = w_obs / w_obs.max()
            V_params = fit_V_params(dx_flat, dy_flat, V_obs, w_obs,
                                    log=log, lambda_theta=lambda_theta)
        else:
            log("V params: мало точек, параметризация невозможна")
            V_params = None
            param_V = False
    else:
        V_params = None

    def get_V_tab():
        if param_V and V_params is not None:
            return V_param(DX, DY, *V_params)
        else:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (dy_c, dx_c), V_grid0, bounds_error=False, fill_value=0.0)
            return interp(np.column_stack([DY + shift_y, DX + shift_x]))

    V_tab = get_V_tab()
    log(f"V_tab: max={V_tab.max():.3f}, mean={V_tab.mean():.3f}")

    # ---------- инициализация ----------
    L = build_laplacian_sparse(PADS_Y, PADS_X) if lam_eps > 0 else None
    n_sat = float(n_sat_init) if fit_sat else 0.0
    dead = dead_init.copy() if dead_init is not None else \
        np.zeros((PADS_Y, PADS_X), dtype=bool)
    if dead.any():
        log(f"dead: стартовая маска = {int(dead.sum())} пикселей")

    eps = np.ones(N_eps)
    b = np.zeros(N_eps)
    if dead.any():
        eps[dead.ravel()] = 0.0
        b[dead.ravel()] = 0.0
    Wf = Wf_base.copy()
    A = solve_A(D, Wf, eps[lin_eps] * V_tab, lin_A, T)

    def loss_N(N_sat):
        model = saturate(eps[lin_eps] * A[lin_A] * V_tab, N_sat) + b[lin_eps]
        return float(np.sum(Wf * (D - model) ** 2))

    # ---------- цикл ----------
    prev = None
    t0 = time.time()
    for it in range(n_iter):
        for irls in range(n_irls if robust else 1):
            D_nb = D - b[lin_eps]
            A = solve_A(desaturate(D_nb, n_sat), Wf,
                        eps[lin_eps] * V_tab, lin_A, T)
            eps = solve_eps(desaturate(D_nb, n_sat), Wf, A[lin_A] * V_tab,
                            lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X,
                            dead=dead)
            if not np.all(np.isfinite(eps)):
                log(f"  it {it:2d}.{irls}: eps содержит nan/inf — прерываем")
                break
            # --- detrend_plane ---
            if detrend:
                eps_2d_tmp = eps.reshape(PADS_Y, PADS_X)
                eps_2d_tmp = detrend_plane(eps_2d_tmp, dead=dead)
                eps = eps_2d_tmp.ravel()
            if (~dead).any():
                m = eps[~dead.ravel()].mean()
            else:
                m = eps.mean()
            m = max(m, 1e-12)
            eps /= m
            if dead.any():
                eps[dead.ravel()] = 0.0
            A *= m
            A = solve_A(desaturate(D_nb, n_sat), Wf,
                        eps[lin_eps] * V_tab, lin_A, T)
            if fit_bg:
                model_no_b = saturate(eps[lin_eps] * A[lin_A] * V_tab, n_sat)
                b = solve_b(D, Wf, model_no_b, lin_eps, N_eps, dead=dead)
            if fit_sat:
                res = minimize_scalar(loss_N, bounds=(1.0, 1e9),
                                      method="bounded",
                                      options={"xatol": 1.0})
                n_sat = float(res.x)

            if robust and irls < (n_irls - 1):
                model = saturate(eps[lin_eps] * A[lin_A] * V_tab, n_sat) + b[lin_eps]
                resid = D - model
                w = robust_weights(resid, c=robust_c)
                Wf = Wf_base * w
            else:
                model = saturate(eps[lin_eps] * A[lin_A] * V_tab, n_sat) + b[lin_eps]
                resid = D - model

        loss_v = float(np.sum(Wf * (D - model) ** 2))
        d_loss = 0.0 if prev is None else abs(loss_v - prev) / max(abs(prev), 1e-30)
        log(f"  it {it:2d}: loss={loss_v:.4e} d_loss={d_loss:.2e}"
            + (f" N_sat={n_sat:.1f}" if fit_sat else "")
            + f" dead={int(dead.sum())}"
            + f" b_mean={b.mean():.3e} b_min={b.min():.3e}"
            + f" time={time.time() - t0:.2f}s")

        if not np.isfinite(loss_v):
            log(f"  it {it:2d}: loss=nan — прерываем")
            break

        eps_2d = eps.reshape(PADS_Y, PADS_X)
        new_dead = detect_dead(eps_2d, None, n_valid_2d,
                               PADS_Y, PADS_X,
                               thresh_eps=dead_thresh_eps,
                               thresh_stat=dead_thresh_stat)
        added = int((new_dead & ~dead).sum())
        if added:
            dead = dead | new_dead
            eps[dead.ravel()] = 0.0
            b[dead.ravel()] = 0.0
            log(f"  it {it:2d}: dead += {added} → {int(dead.sum())}")

        # --- перефит V_params с warmup ---
        if param_V and refit_V:
            if it < warmup_V:
                model_no_V = eps[lin_eps] * A[lin_A]
                m_ok = model_no_V > 1e-6
                if m_ok.sum() > 10:
                    V_obs_pts = D[m_ok] / model_no_V[m_ok]
                    w_pts = Wf[m_ok]
                    V_params = fit_V_params(
                        DX[m_ok], DY[m_ok], V_obs_pts, w_pts,
                        x0=V_params, log=log,
                        lambda_theta=lambda_theta)
                    V_tab = get_V_tab()
                    log(f"  it {it:2d}: V refit (warmup {it+1}/{warmup_V}), "
                        f"theta={np.degrees(V_params[3]):.2f}°")
            elif it == warmup_V:
                log(f"  it {it:2d}: V_params FROZEN at theta="
                    f"{np.degrees(V_params[3]):.2f}°, "
                    f"gamma={V_params[5]:.3f}")

        if prev is not None and d_loss < 1e-6:
            log(f"converged at it {it} (d_loss={d_loss:.2e})")
            break
        prev = loss_v
    log(f"fit time: {time.time() - t0:.2f}s")

    # ---------- финал ----------
    D_nb = D - b[lin_eps]
    A = solve_A(desaturate(D_nb, n_sat), Wf,
                eps[lin_eps] * V_tab, lin_A, T)
    eps = solve_eps(desaturate(D_nb, n_sat), Wf, A[lin_A] * V_tab,
                    lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X, dead=dead)
    if detrend:
        eps_2d_tmp = eps.reshape(PADS_Y, PADS_X)
        eps_2d_tmp = detrend_plane(eps_2d_tmp, dead=dead)
        eps = eps_2d_tmp.ravel()
    if (~dead).any():
        m = eps[~dead.ravel()].mean()
    else:
        m = eps.mean()
    m = max(m, 1e-12)
    eps /= m
    if dead.any():
        eps[dead.ravel()] = 0.0
    A *= m
    A = solve_A(desaturate(D_nb, n_sat), Wf,
                eps[lin_eps] * V_tab, lin_A, T)
    if fit_bg:
        model_no_b = saturate(eps[lin_eps] * A[lin_A] * V_tab, n_sat)
        b = solve_b(D, Wf, model_no_b, lin_eps, N_eps, dead=dead)
    eps_2d = eps.reshape(PADS_Y, PADS_X)
    b_2d = b.reshape(PADS_Y, PADS_X)

    P_t = np.zeros((T, PADS_Y, PADS_X))
    for t in range(T):
        if param_V and V_params is not None:
            pt = A[t] * V_param(dy[t] - shift_y, dx[t] - shift_x, *V_params)
        else:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (dy_c, dx_c), V_grid0, bounds_error=False, fill_value=0.0)
            pt = A[t] * interp(np.column_stack(
                [(dy[t] - shift_y).ravel(), (dx[t] - shift_x).ravel()]))
        P_t[t] = pt.reshape(PADS_Y, PADS_X)

    model = saturate(eps[lin_eps] * A[lin_A] * V_tab, n_sat) + b[lin_eps]
    resid = D - model
    resid_abs = np.abs(resid)
    cnt = np.ones_like(resid, dtype=int)

    return {"eps": eps_2d, "A": A, "P_t": P_t, "N_sat": n_sat,
            "b": b_2d, "fit_bg": fit_bg,
            "V_params": V_params, "param_V": param_V,
            "shift": (shift_x, shift_y),
            "V_tab": V_tab, "dx_c": dx_c, "dy_c": dy_c, "V_grid": V_grid0,
            "dx": dx, "dy": dy, "mask": mask, "dead": dead,
            "Wf": Wf, "Wf_base": Wf_base,
            "resid": resid, "resid_abs": resid_abs, "cnt": cnt,
            "t_idx": t_idx, "i_idx": i_idx, "j_idx": j_idx,
            "D": D, "lin_eps": lin_eps, "lin_A": lin_A}


# ============================================================
#  σ_stat по K фолдам
# ============================================================
def kfold_stat(res, PADS_Y, PADS_X, lam_eps, K=4, detrend=True, log=print):
    T = int(res["lin_A"].max()) + 1
    N_eps = PADS_Y * PADS_X
    D = res["D"]; V_tab = res["V_tab"]
    Wf = res["Wf"] if "Wf" in res else res["W"]
    lin_eps = res["lin_eps"]; lin_A = res["lin_A"]
    n_sat = res["N_sat"]
    dead = res.get("dead", None)
    fit_bg = res.get("fit_bg", False)
    b_init = res["b"].ravel() if "b" in res else None
    D_lin = desaturate(D, n_sat) if n_sat > 0 else D
    L = build_laplacian_sparse(PADS_Y, PADS_X) if lam_eps > 0 else None

    folds = np.full((K, PADS_Y, PADS_X), np.nan)
    for k in range(K):
        sel = (lin_A % K) == k
        eps_k, _, _, _ = _fit_eps_A(
            D_lin[sel], Wf[sel], V_tab[sel],
            lin_eps[sel], lin_A[sel], T, N_eps,
            L, lam_eps, PADS_Y, PADS_X, n_inner=3,
            fit_sat=False, n_sat_init=n_sat, dead=dead,
            fit_bg=fit_bg, b_init=b_init, detrend=detrend)
        if np.all(np.isfinite(eps_k)):
            folds[k] = eps_k
        else:
            log(f"  kfold {k}: WARNING: нефинитный eps")

    eps_stat = np.nanstd(folds, axis=0, ddof=1) * np.sqrt(K - 1)
    n_bad = int(np.sum(~np.isfinite(eps_stat)))
    log(f"kfold({K}): median σ_stat = {np.nanmedian(eps_stat):.3e} "
        f"(нефинитных: {n_bad})")
    return eps_stat, folds


# ============================================================
#  Bootstrap (опционально)
# ============================================================
def bootstrap_full(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
                   lam_eps, step, sigma_smooth, N_BOOT, seed=42, log=print,
                   dead_init=None, n_valid_2d=None,
                   dead_thresh_eps=0.05, dead_thresh_stat=0.01,
                   robust=True, robust_c=3.0, n_irls=2, fit_bg=True,
                   param_V=True, refit_V=True, warmup_V=2,
                   lambda_theta=1.0, detrend=True):
    rng = np.random.default_rng(seed)
    T = data.shape[0]
    eps_boot = np.zeros((N_BOOT, PADS_Y, PADS_X))
    for b in range(N_BOOT):
        idx = rng.integers(0, T, T)
        r = fit_m12(data[idx], data_raw[idx], dx[idx], dy[idx], mask[idx],
                    PADS_Y, PADS_X, n_iter=3, lam_eps=lam_eps,
                    step=step, sigma_smooth=sigma_smooth,
                    fit_sat=False, n_sat_init=0.0,
                    log=lambda *_: None,
                    dead_init=dead_init, n_valid_2d=n_valid_2d,
                    dead_thresh_eps=dead_thresh_eps,
                    dead_thresh_stat=dead_thresh_stat,
                    robust=robust, robust_c=robust_c, n_irls=n_irls,
                    fit_bg=fit_bg, param_V=param_V, refit_V=refit_V,
                    warmup_V=warmup_V, lambda_theta=lambda_theta,
                    detrend=detrend)
        eps_boot[b] = r["eps"]
    eps_stat = np.std(eps_boot, axis=0, ddof=1)
    log(f"bootstrap({N_BOOT}): median σ_stat = {np.nanmedian(eps_stat):.3e}")
    return eps_stat, eps_boot


# ============================================================
#  main
# ============================================================
def main():
    ap = argparse.ArgumentParser(
        description="Восстановление карты эффективности пикселей eps "
                    "с параметрическим V, warmup_V, lambda_theta и detrend.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="data/data20261109_143025.txt",
                    help="Путь к входному файлу кадров")
    ap.add_argument("--real-eps", default="epsilon_real.txt",
                    help="Эталон epsilon.txt для σ_sys (если есть)")
    ap.add_argument("--pads-x", type=int, default=32, help="Ширина, px")
    ap.add_argument("--pads-y", type=int, default=20, help="Высота, px")
    ap.add_argument("--pad-mm", type=float, nargs=2, default=None,
                    help="Размер пикселя в мм (x y)")
    ap.add_argument("--n-iter", type=int, default=10,
                    help="Число внешних итераций")
    ap.add_argument("--lam-eps", type=float, default=0.01,
                    help="Тихоновская регуляризация eps")
    ap.add_argument("--r-max", type=float, default=15.0,
                    help="Макс. радиус маски, px")
    ap.add_argument("--min-valid", type=int, default=200,
                    help="Мин. число валидных кадров")
    ap.add_argument("--r-sigma-max", type=float, default=14.0,
                    help="Макс. радиус для σ_stat, px")
    ap.add_argument("--step", type=float, default=1.0,
                    help="Шаг сетки V_emp (для init)")
    ap.add_argument("--sigma-smooth", type=float, default=0.5,
                    help="σ сглаживания V_emp (для init)")
    ap.add_argument("--kfold", type=int, default=4,
                    help="K фолдов для σ_stat")
    ap.add_argument("--n-boot", type=int, default=0,
                    help=">0: bootstrap")
    ap.add_argument("--fit-sat", action="store_true",
                    help="Фитировать N_sat")
    ap.add_argument("--n-sat-init", type=float, default=0.0,
                    help="Начальное N_sat")
    ap.add_argument("--out", default=".", help="Каталог вывода")

    ap.add_argument("--read-dead", default=None,
                    help="Файл маски мёртвых пикселей")
    ap.add_argument("--save-dead", default=None,
                    help="Куда сохранить маску мёртвых")
    ap.add_argument("--dead-thresh-eps", type=float, default=0.05,
                    help="Порог eps для мёртвых")
    ap.add_argument("--dead-thresh-stat", type=float, default=0.01,
                    help="Порог σ_stat для мёртвых")

    ap.add_argument("--no-robust", dest="robust", action="store_false",
                    help="Отключить IRLS")
    ap.add_argument("--robust-c", type=float, default=3.0,
                    help="Порог Huber")
    ap.add_argument("--n-irls", type=int, default=2,
                    help="Число IRLS-подшагов")

    ap.add_argument("--no-fit-bg", dest="fit_bg", action="store_false",
                    help="Отключить фон b[i,j]")
    ap.add_argument("--no-param-V", dest="param_V", action="store_false",
                    help="Отключить параметрический V (свободная V)")
    ap.add_argument("--no-refit-V", dest="refit_V", action="store_false",
                    help="Не перефичивать V_params на итерациях")

    ap.add_argument("--warmup-V", type=int, default=2,
                    help="Сколько итераций перефитить V_params перед фиксацией "
                         "(0 = фиксировать сразу, -1 = не фиксировать)")
    ap.add_argument("--lambda-theta", type=float, default=1.0,
                    help="Штраф на угол поворота V (слабое притяжение к 0)")
    ap.add_argument("--no-detrend", dest="detrend", action="store_false",
                    help="Отключить detrend_plane (вычитание линейного тренда)")

    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

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
        f"valid pixels = {pix_ok.sum()} / {PADS_Y * PADS_X}")

    # ---------- мёртвые ----------
    dead_init = read_dead_file(args.read_dead, PADS_Y, PADS_X)
    if args.read_dead and os.path.exists(args.read_dead):
        log(f"dead: прочитано {int(dead_init.sum())} пикселей "
            f"из {args.read_dead}")
    else:
        log("dead: стартовая маска пуста")

    # ---------- фит ----------
    log(f"\n=== fit M12 (param_V={args.param_V}, refit_V={args.refit_V}, "
        f"warmup_V={args.warmup_V}, lambda_theta={args.lambda_theta}, "
        f"detrend={args.detrend}, robust={args.robust}, "
        f"fit_bg={args.fit_bg}) ===")
    res = fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
                  n_iter=args.n_iter, lam_eps=args.lam_eps,
                  step=args.step, sigma_smooth=args.sigma_smooth,
                  fit_sat=args.fit_sat, n_sat_init=args.n_sat_init, log=log,
                  dead_init=dead_init, n_valid_2d=n_valid_2d,
                  dead_thresh_eps=args.dead_thresh_eps,
                  dead_thresh_stat=args.dead_thresh_stat,
                  robust=args.robust, robust_c=args.robust_c,
                  n_irls=args.n_irls,
                  fit_bg=args.fit_bg, param_V=args.param_V,
                  refit_V=args.refit_V,
                  warmup_V=args.warmup_V,
                  lambda_theta=args.lambda_theta,
                  detrend=args.detrend)
    eps = res["eps"]; A = res["A"]; n_sat = res["N_sat"]
    dead = res["dead"]; b = res["b"]
    V_params = res["V_params"]; param_V = res["param_V"]
    shift_x, shift_y = res["shift"]
    dx_c = res["dx_c"]; dy_c = res["dy_c"]; V_grid0 = res["V_grid"]

    # ---------- V_dxdy.txt ----------
    DXg, DYg = np.meshgrid(dx_c - shift_x, dy_c - shift_y, indexing="xy")
    if param_V and V_params is not None:
        V_out = V_param(DXg, DYg, *V_params)
        V_out = V_out / V_out.max()
    else:
        V_out = V_grid0 / V_grid0.max()
    np.savetxt(os.path.join(args.out, "V_dxdy.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), V_out.ravel()],
               header="dx dy V")

    # ---------- V_params.txt ----------
    if param_V and V_params is not None:
        with open(os.path.join(args.out, "V_params.txt"), "w") as f:
            f.write("# A sx sy theta_deg eta gamma shift_x shift_y\n")
            f.write(f"{V_params[0]:.10e} {V_params[1]:.10e} "
                    f"{V_params[2]:.10e} {np.degrees(V_params[3]):.6f} "
                    f"{V_params[4]:.10e} {V_params[5]:.10e} "
                    f"{shift_x:.6f} {shift_y:.6f}\n")

    # ---------- V(r) model vs V_emp ----------
    t_idx = res["t_idx"]; i_idx = res["i_idx"]; j_idx = res["j_idx"]
    D = res["D"]
    DX = dx[t_idx, i_idx, j_idx] - shift_x
    DY = dy[t_idx, i_idx, j_idx] - shift_y
    R_pt = np.sqrt(DX ** 2 + DY ** 2)
    r_bins = np.linspace(0, args.r_max, 31)
    H, _ = np.histogram(R_pt, bins=r_bins, weights=D)
    C, _ = np.histogram(R_pt, bins=r_bins)
    V_emp_r = np.where(C > 0, H / np.maximum(C, 1e-9), 0.0)
    cb = 0.5 * (r_bins[1:] + r_bins[:-1])

    Rg = np.sqrt(DXg ** 2 + DYg ** 2)
    V_r_model = np.zeros(len(r_bins) - 1)
    for k in range(len(r_bins) - 1):
        m = (Rg >= r_bins[k]) & (Rg < r_bins[k + 1]) & (V_out > 0)
        if m.sum() > 0:
            V_r_model[k] = V_out[m].mean()
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

    # ---------- диагностика остатков ----------
    V_tab = res["V_tab"]
    Wf = res["Wf"]
    resid = res["resid"]
    abs_r = res["resid_abs"]

    rms = float(np.sqrt(np.mean(resid ** 2)))
    sigma_mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    frac_3s = float(np.mean(abs_r > 3 * rms))
    frac_5s = float(np.mean(abs_r > 5 * rms))
    frac_3mad = float(np.mean(abs_r > 3 * sigma_mad))
    frac_5mad = float(np.mean(abs_r > 5 * sigma_mad))
    chi2_total = float(np.sum(Wf * resid ** 2))
    chi2_out5 = float(np.sum(Wf[abs_r > 5 * rms] * resid[abs_r > 5 * rms] ** 2))
    chi2_out5_frac = chi2_out5 / max(chi2_total, 1e-30)
    w_ratio = Wf / np.maximum(res["Wf_base"], 1e-30)
    log(f"\n=== диагностика остатков ===")
    log(f"rms = {rms:.4e}, max|r| = {np.max(abs_r):.4e}, "
        f"sigma_MAD = {sigma_mad:.4e}")
    log(f"frac(|r|>3·rms) = {frac_3s:.3%}, "
        f"frac(|r|>5·rms) = {frac_5s:.3%}")
    log(f"frac(|r|>3·σ_MAD) = {frac_3mad:.3%}, "
        f"frac(|r|>5·σ_MAD) = {frac_5mad:.3%}")
    log(f"χ²_w/M = {chi2_total / D.size:.4e}")
    log(f"χ²: total = {chi2_total:.4e}, "
        f"outliers(|r|>5·rms) = {chi2_out5:.4e} "
        f"({100 * chi2_out5_frac:.1f}% от χ²)")
    log(f"Wf/Wf_base: median = {np.median(w_ratio):.3f}, "
        f"min = {np.min(w_ratio):.3e}")
    log(f"b: mean = {b.mean():.4e}, max = {b.max():.4e}, "
        f"min = {b.min():.4e}, median = {np.median(b):.4e}")

    # ---------- карта остатков ----------
    resid_sum = np.zeros((PADS_Y, PADS_X))
    abs_sum = np.zeros((PADS_Y, PADS_X))
    cnt_2d = np.zeros((PADS_Y, PADS_X), dtype=int)
    np.add.at(resid_sum, (i_idx, j_idx), resid)
    np.add.at(abs_sum, (i_idx, j_idx), abs_r)
    np.add.at(cnt_2d, (i_idx, j_idx), 1)
    mean_resid = np.where(cnt_2d > 0, resid_sum / np.maximum(cnt_2d, 1), np.nan)
    mean_abs = np.where(cnt_2d > 0, abs_sum / np.maximum(cnt_2d, 1), np.nan)

    np.save(os.path.join(args.out, "resid_mean.npy"), mean_resid)
    np.save(os.path.join(args.out, "resid_abs_mean.npy"), mean_abs)
    np.save(os.path.join(args.out, "resid_counts.npy"), cnt_2d)

    with open(os.path.join(args.out, "resid_map.txt"), "w") as f:
        f.write("# x y mean_resid mean_abs_resid n_points\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {mean_resid[i,j]:.6e} "
                        f"{mean_abs[i,j]:.6e} {cnt_2d[i,j]}\n")

    # ---------- топ-10 ----------
    flat_order = np.argsort(np.nan_to_num(mean_abs, nan=-1).ravel())[::-1]
    log("\n=== топ-10 пикселей по mean|resid| ===")
    log(f"{'x':>4s} {'y':>4s} {'mean|r|':>12s} {'mean_r':>12s} "
        f"{'n':>6s} {'eps':>10s} {'b':>10s} {'dead':>4s}")
    for k in flat_order[:10]:
        i, j = divmod(int(k), PADS_X)
        log(f"{j:4d} {i:4d} {mean_abs[i,j]:12.4e} {mean_resid[i,j]:12.4e} "
            f"{cnt_2d[i,j]:6d} {eps[i,j]:10.3e} {b[i,j]:10.3e} "
            f"{1 if dead[i,j] else 0:4d}")

    flat_bias = np.argsort(np.abs(np.nan_to_num(mean_resid, nan=0)).ravel())[::-1]
    log("\n=== топ-10 пикселей по |mean_resid| (смещение) ===")
    for k in flat_bias[:10]:
        i, j = divmod(int(k), PADS_X)
        log(f"{j:4d} {i:4d} mean_r={mean_resid[i,j]:12.4e} "
            f"mean|r|={mean_abs[i,j]:12.4e} n={cnt_2d[i,j]:6d} "
            f"eps={eps[i,j]:10.3e} b={b[i,j]:10.3e}")

    # ---------- σ_stat ----------
    if args.n_boot > 0:
        log(f"\n=== σ_stat (bootstrap {args.n_boot}) ===")
        eps_stat, folds = bootstrap_full(
            data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
            lam_eps=args.lam_eps, step=args.step,
            sigma_smooth=args.sigma_smooth,
            N_BOOT=args.n_boot, log=log,
            dead_init=dead, n_valid_2d=n_valid_2d,
            dead_thresh_eps=args.dead_thresh_eps,
            dead_thresh_stat=args.dead_thresh_stat,
            robust=args.robust, robust_c=args.robust_c,
            n_irls=args.n_irls, fit_bg=args.fit_bg,
            param_V=args.param_V, refit_V=args.refit_V,
            warmup_V=args.warmup_V, lambda_theta=args.lambda_theta,
            detrend=args.detrend)
    elif args.kfold > 1:
        log(f"\n=== σ_stat (kfold K={args.kfold}) ===")
        eps_stat, folds = kfold_stat(
            res, PADS_Y, PADS_X, lam_eps=args.lam_eps,
            K=args.kfold, detrend=args.detrend, log=log)
    else:
        log("\n=== σ_stat (split-half) ===")
        eps_stat, folds = kfold_stat(
            res, PADS_Y, PADS_X, lam_eps=args.lam_eps, K=2,
            detrend=args.detrend, log=log)

    r_2d = np.sqrt(
        (np.arange(PADS_X, dtype=float)[None, :] - CX) ** 2 +
        (np.arange(PADS_Y, dtype=float)[:, None] - CY) ** 2
    )
    eps_stat[r_2d > args.r_sigma_max] = np.nan
    log(f"σ_stat = nan для r > {args.r_sigma_max}: "
        f"{(r_2d > args.r_sigma_max).sum()} пикселей")

    # ---------- финальное уточнение маски ----------
    dead_final = dead | detect_dead(eps, eps_stat, n_valid_2d,
                                    PADS_Y, PADS_X,
                                    thresh_eps=args.dead_thresh_eps,
                                    thresh_stat=args.dead_thresh_stat)
    if int(dead_final.sum()) != int(dead.sum()):
        log(f"dead: уточнено по σ_stat: {int(dead.sum())} → "
            f"{int(dead_final.sum())}")
        eps[dead_final] = 0.0
        b[dead_final] = 0.0
        eps_stat[dead_final] = np.nan
        dead = dead_final

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
    with open(os.path.join(out, "epsilon.txt"), "w") as f:
        f.write("# x y eps sigma_stat sigma_sys sigma_tot dead_flag b\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {eps[i,j]:.10e} {eps_stat[i,j]:.6e} "
                        f"{eps_sys[i,j]:.6e} {eps_tot[i,j]:.6e} "
                        f"{1 if dead[i,j] else 0} {b[i,j]:.10e}\n")

    P_avg = res["P_t"].mean(axis=0)
    with open(os.path.join(out, "avg_data.txt"), "w") as f:
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {P_avg[i,j]:.10e}\n")
    np.save(os.path.join(out, "avg_data.npy"), P_avg)

    np.savetxt(os.path.join(out, "V_r.txt"),
               np.c_[cb, V_emp_r, V_r_model * scale],
               header="r V_emp V_model_scaled")

    A_mean = A.mean()
    V_amp = V_out * A_mean
    np.savetxt(os.path.join(out, "V_dxdy_amp.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), V_amp.ravel()],
               header="dx dy F")

    np.save(os.path.join(out, "bkg.npy"), b)
    np.savetxt(os.path.join(out, "bkg.txt"),
               np.c_[np.indices((PADS_Y, PADS_X))[1].ravel(),
                     np.indices((PADS_Y, PADS_X))[0].ravel(),
                     b.ravel()],
               header="x y b")

    np.save(os.path.join(out, "P_t.npy"), res["P_t"])
    np.save(os.path.join(out, "eps.npy"), eps)
    np.save(os.path.join(out, "A.npy"), A)
    np.save(os.path.join(out, "eps_folds.npy"), folds)
    np.save(os.path.join(out, "N_sat.npy"), np.array(n_sat))

    if args.save_dead:
        write_dead_file(args.save_dead, dead, PADS_Y, PADS_X)
        log(f"dead: сохранено {int(dead.sum())} пикселей в {args.save_dead}")

    # ---------- σ пучка ----------
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
    if param_V and V_params is not None:
        log(f"\n=== V параметры (повёрнутый псевдо-Voigt) ===")
        log(f"A = {V_params[0]:.6e}")
        log(f"σx = {V_params[1]:.4f} px")
        log(f"σy = {V_params[2]:.4f} px")
        log(f"θ = {np.degrees(V_params[3]):.3f}°")
        log(f"η (доля Коши) = {V_params[4]:.4f}")
        log(f"γ (ширина Коши) = {V_params[5]:.4f}")
        log(f"центр: dx = {shift_x:.3f}, dy = {shift_y:.3f}")

    with open(os.path.join(out, "fit_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
