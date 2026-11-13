#!/usr/bin/env python3
"""
get_epsilon.py -- восстановление карты эффективности пикселей eps
по кадрам со сканирующим пучком.

Модель:
    data[t,i,j] ~= sat( eps[i,j] * A_t * V(dx, dy), N_sat ) + b[i,j]
    dx = j - c_x[t] - center_offset_x
    dy = i - c_y[t] - center_offset_y

V(dx,dy) -- параметрический повернутый псевдо-Voigt ИЛИ сумма 2 Voigt,
либо свободная V_emp (интерполяция).

Режимы (--eps-mode):
  * strict (по умолчанию, финальный):
        никаких крупномасштабных структур (ни ямки, ни градиента).
        Вся крупномасштабная структура убирается через high_pass
        с sigma=2.0: eps_hp = eps - gaussian_filter(eps, 2.0) + mean.
        Остаются только локальные дефекты (размер < 2 пикселей).
  * physical:
        устойчивые крупномасштабные структуры РАЗРЕШЕНЫ
        (ямка/виньетирование остаются в eps, артефактные градиенты
        убираются detrend_periphery: поверхность подбирается только
        по периферии, центр не трогается).
        Для этого детектора НЕ рекомендуется (ямка -- глобальный тренд).

Финальный вариант (strict, sigma=2, r_max=13):
    python get_epsilon.py --data <file> --out <dir>

    Все параметры уже выставлены как дефолты:
      --eps-mode strict
      --high-pass-sigma 2.0
      --lam-eps 0.01
      --r-max 13
      --n-iter 15
      --center-offset-mode smooth-argmax
      --zero-weight-mode one
      --detrend-order None (-> high_pass)
      --param-V False (свободная V)
      --real-eps "" (эталон не используется)

    Результат: eps с corr ~ 0.56 и rms ~ 0.05 на двух наборах данных
    (в пределах r <= 13 от центра). Локальные дефекты сохранены,
    глобальные тренды убраны.

Ключевые решения:
  * data нормируется на ток I (I0 = median(I)).
  * warmup_V: V_params фитируются первые N итераций, потом фиксируются.
  * lambda_theta: слабый штраф на угол.
  * gamma_min: нижняя граница gamma.
  * detrend: none|plane|quadratic|periphery_plane|periphery_quadratic|high_pass.
  * center-offset: none|auto|manual|smooth-argmax.
  * Closed-form eps, A_t, b.
  * Тихонов по eps (лапласиан) или без него (lam_eps=0).
  * sigma_stat по K=4 фолдам (std * sqrt(K-1)).
  * Насыщение sat(., N_sat) опционально.
  * Мёртвые пиксели:
      - dead-from-data: нуль во ВСЕХ кадрах -> dead (всегда);
      - авто-детекция по eps: только если --dead-thresh-eps > 0.
  * W_base = 1/max(data, 1) для всех пикселей.
      --zero-weight-mode zero : вес 1 (нули -- сигнал, для Poisson)
      --zero-weight-mode one  : вес 0 (нули -- выпадения, для реальных)
  * Робастные веса (Huber) с защитой от MAD=0.
  * Защита fit_V_params: A ограничен 10xV_obs.max(), откат при развале.
  * Аддитивный фон b[i,j].
  * Диагностика: center_diag, zero_diag, center_smooth, V_emp_dxdy,
    остатки, карты, топ-10.

Выход: epsilon.txt, avg_data.txt/.npy, V_params.txt, V_dxdy.txt,
       V_dxdy_amp.txt, V_emp_dxdy.txt, V_r.txt, bkg.npy, bkg.txt,
       P_t.npy, eps.npy, A.npy, eps_folds.npy, N_sat.npy,
       fit_log.txt, resid_mean.npy, resid_abs_mean.npy,
       resid_counts.npy, resid_map.txt, center_diag.txt, zero_diag.txt
       (+ center_smooth.txt при smooth-argmax, + маска если --save-dead).
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


#                            saturation
def saturate(true, N_sat):
    if N_sat <= 0:
        return true
    return true / (1.0 + true / N_sat)


def desaturate(obs, N_sat):
    if N_sat <= 0:
        return obs
    return obs / np.maximum(1.0 - obs / N_sat, 1e-6)


#                            geometry
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


#                       parametric V profile
def _pseudo_voigt(u, v, eta, gamma):
    g = np.exp(-0.5 * (u * u + v * v))
    l = 1.0 / (1.0 + u * u + v * v + gamma * gamma)
    return eta * l + (1.0 - eta) * g


def _gauss(u, v):
    return np.exp(-0.5 * (u * u + v * v))


def V_param(dx, dy, A, sx, sy, theta, eta, gamma):
    ct, st = np.cos(theta), np.sin(theta)
    u = (dx * ct + dy * st) / sx
    v = (-dx * st + dy * ct) / sy
    return A * _pseudo_voigt(u, v, eta, gamma)


def V_param2(dx, dy, A, sx1, sy1, theta, eta, gamma, sx2, sy2, w):
    ct, st = np.cos(theta), np.sin(theta)
    u1 = (dx * ct + dy * st) / sx1
    v1 = (-dx * st + dy * ct) / sy1
    u2 = (dx * ct + dy * st) / sx2
    v2 = (-dx * st + dy * ct) / sy2
    voigt = _pseudo_voigt(u1, v1, eta, gamma)
    gauss = _gauss(u2, v2)
    return A * (w * voigt + (1.0 - w) * gauss)


def fit_V_params(dx_flat, dy_flat, V_obs, w_obs,
                 x0=None, bounds=None, log=print,
                 lambda_theta=1.0, lambda_gamma=0.0, gamma_min=0.1,
                 sum2v=True):
    if sum2v:
        if x0 is None:
            x0 = np.array([V_obs.max(), 3.0, 3.0, 0.0, 0.5, 0.5,
                           8.0, 8.0, 0.5])
        # A ограничен 10x наблюдаемого максимума -- откат при развале
        A_max = max(float(V_obs.max()) * 10.0, 10.0)
        if bounds is None:
            bounds = (
                [0.0, 0.5, 0.5, -np.pi, 0.0, gamma_min, 0.5, 0.5, 0.0],
                [A_max, 20.0, 20.0, np.pi, 1.0, 10.0, 30.0, 30.0, 1.0]
            )

        def resid(p):
            A, sx1, sy1, th, eta, gam, sx2, sy2, w = p
            V_pred = V_param2(dx_flat, dy_flat, A, sx1, sy1, th, eta,
                              gam, sx2, sy2, w)
            r = w_obs * (V_obs - V_pred)
            pen_theta = np.sqrt(lambda_theta) * th
            if lambda_gamma > 0:
                pen_gamma = np.sqrt(lambda_gamma) * (gam - 0.5)
                return np.r_[r, pen_theta, pen_gamma]
            return np.r_[r, pen_theta]

        try:
            res = least_squares(resid, x0, bounds=bounds, method="trf",
                                max_nfev=3000, xtol=1e-10, ftol=1e-10,
                                gtol=1e-10)
            A, sx1, sy1, th, eta, gam, sx2, sy2, w = res.x
            V_pred = V_param2(dx_flat, dy_flat, A, sx1, sy1, th, eta,
                              gam, sx2, sy2, w)
            rms_resid = float(np.sqrt(np.mean((w_obs * (V_obs - V_pred)) ** 2)))
            V_pred_0 = V_param2(dx_flat, dy_flat, *x0)
            rms_0 = float(np.sqrt(np.mean((w_obs * (V_obs - V_pred_0)) ** 2)))
            if not np.isfinite(rms_resid) or rms_resid > 10.0 * max(rms_0, 1e-12):
                log(f"  V params fit (sum2v): REJECT (rms {rms_resid:.3e} > "
                    f"10x{rms_0:.3e}), откат к x0")
                return x0
            log(f"  V params fit (sum2v): cost={res.cost:.3e}, "
                f"rms_resid={rms_resid:.4e}, A={A:.3f}, "
                f"sx1={sx1:.2f}, sy1={sy1:.2f}, theta={np.degrees(th):.2f}deg, "
                f"eta={eta:.3f}, gamma={gam:.3f}, "
                f"sx2={sx2:.2f}, sy2={sy2:.2f}, w={w:.3f}")
            if gam <= gamma_min + 1e-4:
                log(f"    WARNING: gamma at lower bound ({gamma_min})")
            if sx1 >= 19.99 or sy1 >= 19.99:
                log(f"    WARNING: sigma1 at upper bound (20)")
            if sx2 >= 29.99 or sy2 >= 29.99:
                log(f"    WARNING: sigma2 at upper bound (30)")
            return res.x
        except Exception as e:
            log(f"  V params fit (sum2v) FAILED: {e}")
            return x0
    else:
        if x0 is None:
            x0 = np.array([V_obs.max(), 3.0, 3.0, 0.0, 0.5, 0.5])
        A_max = max(float(V_obs.max()) * 10.0, 10.0)
        if bounds is None:
            bounds = (
                [0.0, 0.5, 0.5, -np.pi, 0.0, gamma_min],
                [A_max, 20.0, 20.0, np.pi, 1.0, 10.0]
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
                                max_nfev=2000, xtol=1e-10, ftol=1e-10,
                                gtol=1e-10)
            A, sx, sy, th, eta, gam = res.x
            V_pred = V_param(dx_flat, dy_flat, A, sx, sy, th, eta, gam)
            rms_resid = float(np.sqrt(np.mean((w_obs * (V_obs - V_pred)) ** 2)))
            V_pred_0 = V_param(dx_flat, dy_flat, *x0)
            rms_0 = float(np.sqrt(np.mean((w_obs * (V_obs - V_pred_0)) ** 2)))
            if not np.isfinite(rms_resid) or rms_resid > 10.0 * max(rms_0, 1e-12):
                log(f"  V params fit: REJECT (rms {rms_resid:.3e} > "
                    f"10x{rms_0:.3e}), откат к x0")
                return x0
            log(f"  V params fit: cost={res.cost:.3e}, "
                f"rms_resid={rms_resid:.4e}, A={A:.3f}, sx={sx:.2f}, "
                f"sy={sy:.2f}, theta={np.degrees(th):.2f}deg, "
                f"eta={eta:.3f}, gamma={gam:.3f}")
            if gam <= gamma_min + 1e-4:
                log(f"    WARNING: gamma at lower bound ({gamma_min})")
            if sx >= 19.99 or sy >= 19.99:
                log(f"    WARNING: sigma at upper bound (20)")
            return res.x
        except Exception as e:
            log(f"  V params fit FAILED: {e}")
            return x0


#                       empirical V profile
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
    # заливка "дыр" итеративно: 4-соседняя диффузия известных значений
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


#                       sparse Laplacian (Neumann)
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


#                       dead pixels
def detect_dead(eps, eps_stat, n_valid_2d, PADS_Y, PADS_X,
                thresh_eps, thresh_stat):
    if thresh_eps <= 0.0:
        return np.zeros((PADS_Y, PADS_X), dtype=bool)
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


#                       closed-form eps, A_t, b
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


#                       robust weights (Huber)
def robust_weights(resid, c=3.0):
    r = np.asarray(resid, dtype=float)
    med = np.median(r)
    mad = np.median(np.abs(r - med))
    sigma = 1.4826 * mad
    # защита от MAD=0 (все остатки одинаковые) -- иначе деление на 0
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.ones_like(r)
    r_scaled = np.abs(r - med) / sigma
    w = np.where(r_scaled <= c, 1.0, c / np.maximum(r_scaled, 1e-12))
    w = np.maximum(w, 0.01)
    return w


#                       detrend: plane, quadratic, periphery_*, high_pass
def detrend_plane(eps_2d, dead=None):
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


def detrend_quadratic(eps_2d, dead=None):
    Y, X = eps_2d.shape
    yy, xx = np.mgrid[0:Y, 0:X]
    xn = (xx - (X - 1) / 2.0) / max((X - 1) / 2.0, 1e-9)
    yn = (yy - (Y - 1) / 2.0) / max((Y - 1) / 2.0, 1e-9)
    A = np.c_[np.ones(Y * X), xn.ravel(), yn.ravel(),
              (xn ** 2).ravel(), (yn ** 2).ravel(), (xn * yn).ravel()]
    y = eps_2d.ravel().copy()
    if dead is not None and dead.any():
        mask_alive = ~dead.ravel()
        if mask_alive.sum() > 6:
            coef, *_ = np.linalg.lstsq(A[mask_alive], y[mask_alive], rcond=None)
        else:
            coef = np.array([1.0, 0, 0, 0, 0, 0])
    else:
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    surface = (A @ coef).reshape(Y, X)
    eps_detrended = eps_2d - surface + surface.mean()
    eps_detrended = np.maximum(eps_detrended, 0.0)
    if dead is not None and dead.any():
        eps_detrended[dead] = 0.0
    return eps_detrended


def detrend_periphery(eps_2d, center_x, center_y, order="plane", dead=None):
    Y, X = eps_2d.shape
    yy, xx = np.mgrid[0:Y, 0:X]
    center_mask = np.zeros((Y, X), dtype=bool)
    cx0, cx1 = center_x
    cy0, cy1 = center_y
    center_mask[cy0:cy1 + 1, cx0:cx1 + 1] = True
    fit_mask = ~center_mask
    if dead is not None:
        fit_mask &= ~dead
    if fit_mask.sum() < 10:
        return eps_2d

    if order == "quadratic":
        xn = (xx - (X - 1) / 2.0) / max((X - 1) / 2.0, 1e-9)
        yn = (yy - (Y - 1) / 2.0) / max((Y - 1) / 2.0, 1e-9)
        A = np.c_[np.ones(Y * X), xn.ravel(), yn.ravel(),
                  (xn ** 2).ravel(), (yn ** 2).ravel(), (xn * yn).ravel()]
    else:
        A = np.c_[np.ones(Y * X), xx.ravel(), yy.ravel()]

    y = eps_2d.ravel().copy()
    mask = fit_mask.ravel()
    coef, *_ = np.linalg.lstsq(A[mask], y[mask], rcond=None)
    surface = (A @ coef).reshape(Y, X)
    surface_mean = surface[fit_mask].mean()
    eps_detrended = eps_2d - surface + surface_mean
    eps_detrended = np.maximum(eps_detrended, 0.0)
    if dead is not None and dead.any():
        eps_detrended[dead] = 0.0
    return eps_detrended


def high_pass_eps(eps_2d, sigma=2.0, dead=None):
    from scipy.ndimage import gaussian_filter
    eps_smooth = gaussian_filter(eps_2d, sigma=sigma)
    eps_hp = eps_2d - eps_smooth + eps_smooth.mean()
    eps_hp = np.maximum(eps_hp, 0.0)
    if dead is not None and dead.any():
        eps_hp[dead] = 0.0
    return eps_hp


def detrend(eps_2d, order="plane", dead=None,
            center_x=None, center_y=None, high_pass_sigma=2.0):
    if order == "quadratic":
        return detrend_quadratic(eps_2d, dead=dead)
    elif order == "plane":
        return detrend_plane(eps_2d, dead=dead)
    elif order == "periphery_plane":
        if center_x is None or center_y is None:
            raise ValueError("periphery_plane требует center_x, center_y")
        return detrend_periphery(eps_2d, center_x, center_y,
                                 order="plane", dead=dead)
    elif order == "periphery_quadratic":
        if center_x is None or center_y is None:
            raise ValueError("periphery_quadratic требует center_x, center_y")
        return detrend_periphery(eps_2d, center_x, center_y,
                                 order="quadratic", dead=dead)
    elif order == "high_pass":
        return high_pass_eps(eps_2d, sigma=high_pass_sigma, dead=dead)
    elif order == "none":
        return eps_2d
    else:
        raise ValueError(f"Unknown detrend order: {order}")


#                       one eps/A/b fit step
def _fit_eps_A(D, Wf, V_tab, lin_eps, lin_A, T, N_eps,
               L, lam_eps, PADS_Y, PADS_X,
               n_inner=3, fit_sat=False, n_sat_init=0.0, dead=None,
               fit_bg=True, b_init=None, detrend_order="plane",
               center_x=None, center_y=None, high_pass_sigma=2.0):
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
        if detrend_order != "none":
            eps_2d_tmp = eps.reshape(PADS_Y, PADS_X)
            eps_2d_tmp = detrend(eps_2d_tmp, order=detrend_order, dead=dead,
                                 center_x=center_x, center_y=center_y,
                                 high_pass_sigma=high_pass_sigma)
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


#                       main fit
def fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
            n_iter=3, lam_eps=0.01, step=1.0, sigma_smooth=1.0,
            fit_sat=False, n_sat_init=0.0, log=print,
            dead_init=None, n_valid_2d=None,
            dead_thresh_eps=0.05, dead_thresh_stat=0.01,
            robust=True, robust_c=3.0, n_irls=2,
            fit_bg=True, param_V=True, refit_V=True,
            warmup_V=2, lambda_theta=1.0,
            detrend_order="plane", gamma_min=0.1,
            sum2v=True, zero_weight_mode="zero",
            center_x=None, center_y=None, high_pass_sigma=2.0):
    T = data.shape[0]
    N_eps = PADS_Y * PADS_X
    # W_base: 1/max(data,1); в режиме one нули -> вес 0 (выпадения)
    if zero_weight_mode == "zero":
        W_base = 1.0 / np.maximum(data_raw, 1.0)
    elif zero_weight_mode == "one":
        W_base = np.where(data_raw > 0.0,
                          1.0 / np.maximum(data_raw, 1.0),
                          0.0)
    else:
        raise ValueError(f"Unknown zero_weight_mode: {zero_weight_mode}")

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

    dx_c, dy_c, V_grid0, C_grid = empirical_V_2d(
        dx, dy, data, mask, step=step, sigma_smooth=sigma_smooth)
    log(f"V_emp 2D (init): shape={V_grid0.shape}, "
        f"bins with data: {(C_grid > 0).sum()} / {C_grid.size}")

    # сдвиг профиля V: центр = максимум эмпирической V
    iy_max, ix_max = np.unravel_index(np.argmax(V_grid0), V_grid0.shape)
    shift_x = dx_c[ix_max]
    shift_y = dy_c[iy_max]
    log(f"V center offset: dx={shift_x:.2f}, dy={shift_y:.2f}")

    dx = dx - shift_x
    dy = dy - shift_y
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]

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
                                    log=log, lambda_theta=lambda_theta,
                                    gamma_min=gamma_min, sum2v=sum2v)
        else:
            log("V params: мало точек, параметризация невозможна")
            V_params = None
            param_V = False
    else:
        V_params = None

    def get_V_tab():
        if param_V and V_params is not None:
            if sum2v:
                return V_param2(DX, DY, *V_params)
            else:
                return V_param(DX, DY, *V_params)
        else:
            from scipy.interpolate import RegularGridInterpolator
            interp = RegularGridInterpolator(
                (dy_c, dx_c), V_grid0, bounds_error=False, fill_value=0.0)
            return interp(np.column_stack([DY + shift_y, DX + shift_x]))

    V_tab = get_V_tab()
    log(f"V_tab: max={V_tab.max():.3f}, mean={V_tab.mean():.3f}")

    L = build_laplacian_sparse(PADS_Y, PADS_X) if lam_eps > 0 else None
    n_sat = float(n_sat_init) if fit_sat else 0.0
    dead = dead_init.copy() if dead_init is not None else \
        np.zeros((PADS_Y, PADS_X), dtype=bool)
    if dead.any():
        log(f"dead: стартовая маска = {int(dead.sum())} пикселей")
    else:
        log("dead: стартовая маска пуста")

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
                log(f"  it {it:2d}.{irls}: eps содержит nan/inf -- прерываем")
                break
            if detrend_order != "none":
                eps_2d_tmp = eps.reshape(PADS_Y, PADS_X)
                eps_2d_tmp = detrend(eps_2d_tmp, order=detrend_order, dead=dead,
                                     center_x=center_x, center_y=center_y,
                                     high_pass_sigma=high_pass_sigma)
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
            log(f"  it {it:2d}: loss=nan -- прерываем")
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
            log(f"  it {it:2d}: dead += {added} -> {int(dead.sum())}")
        if param_V and refit_V:
            if warmup_V < 0 or it < warmup_V:
                model_no_V = eps[lin_eps] * A[lin_A]
                m_ok = model_no_V > 1e-6
                if m_ok.sum() > 10:
                    V_obs_pts = D[m_ok] / model_no_V[m_ok]
                    w_pts = Wf[m_ok]
                    V_params = fit_V_params(
                        DX[m_ok], DY[m_ok], V_obs_pts, w_pts,
                        x0=V_params, log=log,
                        lambda_theta=lambda_theta,
                        gamma_min=gamma_min, sum2v=sum2v)
                    V_tab = get_V_tab()
                    log(f"  it {it:2d}: V refit (warmup "
                        f"{'inf' if warmup_V < 0 else f'{it+1}/{warmup_V}'})")
            elif it == warmup_V:
                log(f"  it {it:2d}: V_params FROZEN")
        if prev is not None and d_loss < 1e-6:
            log(f"converged at it {it} (d_loss={d_loss:.2e})")
            break
        prev = loss_v
    log(f"fit time: {time.time() - t0:.2f}s")

    D_nb = D - b[lin_eps]
    A = solve_A(desaturate(D_nb, n_sat), Wf,
                eps[lin_eps] * V_tab, lin_A, T)
    eps = solve_eps(desaturate(D_nb, n_sat), Wf, A[lin_A] * V_tab,
                    lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X, dead=dead)
    if detrend_order != "none":
        eps_2d_tmp = eps.reshape(PADS_Y, PADS_X)
        eps_2d_tmp = detrend(eps_2d_tmp, order=detrend_order, dead=dead,
                             center_x=center_x, center_y=center_y,
                             high_pass_sigma=high_pass_sigma)
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

    # P_t: профиль пучка на каждом кадре, без eps и b
    P_t = np.zeros((T, PADS_Y, PADS_X))
    for t in range(T):
        if param_V and V_params is not None:
            if sum2v:
                pt = A[t] * V_param2(dy[t] - shift_y, dx[t] - shift_x, *V_params)
            else:
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
            "V_params": V_params, "param_V": param_V, "sum2v": sum2v,
            "shift": (shift_x, shift_y),
            "V_tab": V_tab, "dx_c": dx_c, "dy_c": dy_c, "V_grid": V_grid0,
            "dx": dx, "dy": dy, "mask": mask, "dead": dead,
            "Wf": Wf, "Wf_base": Wf_base,
            "resid": resid, "resid_abs": resid_abs, "cnt": cnt,
            "t_idx": t_idx, "i_idx": i_idx, "j_idx": j_idx,
            "D": D, "lin_eps": lin_eps, "lin_A": lin_A}


#                       sigma_stat via K folds
def kfold_stat(res, PADS_Y, PADS_X, lam_eps, K=4,
               detrend_order="plane", log=print,
               center_x=None, center_y=None, high_pass_sigma=2.0):
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
            fit_bg=fit_bg, b_init=b_init, detrend_order=detrend_order,
            center_x=center_x, center_y=center_y,
            high_pass_sigma=high_pass_sigma)
        if np.all(np.isfinite(eps_k)):
            folds[k] = eps_k
        else:
            log(f"  kfold {k}: WARNING: нефинитный eps")
    # sigma_stat = std(folds) * sqrt(K-1) -- поправка на K фолдов
    eps_stat = np.nanstd(folds, axis=0, ddof=1) * np.sqrt(K - 1)
    n_bad = int(np.sum(~np.isfinite(eps_stat)))
    log(f"kfold({K}): median sigma_stat = {np.nanmedian(eps_stat):.3e} "
        f"(нефинитных: {n_bad})")
    return eps_stat, folds


#                       bootstrap (optional)
def bootstrap_full(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
                   lam_eps, step, sigma_smooth, N_BOOT, seed=42, log=print,
                   dead_init=None, n_valid_2d=None,
                   dead_thresh_eps=0.05, dead_thresh_stat=0.01,
                   robust=True, robust_c=3.0, n_irls=2, fit_bg=True,
                   param_V=True, refit_V=True, warmup_V=2,
                   lambda_theta=1.0, detrend_order="plane", gamma_min=0.1,
                   sum2v=True, zero_weight_mode="zero",
                   center_x=None, center_y=None, high_pass_sigma=2.0):
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
                    detrend_order=detrend_order, gamma_min=gamma_min,
                    sum2v=sum2v, zero_weight_mode=zero_weight_mode,
                    center_x=center_x, center_y=center_y,
                    high_pass_sigma=high_pass_sigma)
        eps_boot[b] = r["eps"]
    eps_stat = np.std(eps_boot, axis=0, ddof=1)
    log(f"bootstrap({N_BOOT}): median sigma_stat = {np.nanmedian(eps_stat):.3e}")
    return eps_stat, eps_boot


#                            main
def main():
    ap = argparse.ArgumentParser(
        description="Восстановление карты eps. "
                    "По умолчанию: strict (high_pass sigma=2, r_max=13), "
                    "smooth-argmax, zero-weight-mode=one, свободная V.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--data", default="data/data20261109_143025.txt")
    ap.add_argument("--real-eps", default="",
                    help="Эталон epsilon.txt для sigma_sys (если есть). "
                         "По умолчанию пусто -- не искать.")
    ap.add_argument("--pads-x", type=int, default=32)
    ap.add_argument("--pads-y", type=int, default=20)
    ap.add_argument("--pad-mm", type=float, nargs=2, default=None)
    ap.add_argument("--n-iter", type=int, default=15)
    ap.add_argument("--lam-eps", type=float, default=0.01)
    ap.add_argument("--r-max", type=float, default=13.0)
    ap.add_argument("--min-valid", type=int, default=200)
    ap.add_argument("--r-sigma-max", type=float, default=14.0)
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--sigma-smooth", type=float, default=0.5)
    ap.add_argument("--kfold", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--fit-sat", action="store_true")
    ap.add_argument("--n-sat-init", type=float, default=0.0)
    ap.add_argument("--out", default=".")
    ap.add_argument("--read-dead", default=None)
    ap.add_argument("--save-dead", default=None)
    ap.add_argument("--dead-thresh-eps", type=float, default=0.05)
    ap.add_argument("--dead-thresh-stat", type=float, default=0.01)
    ap.add_argument("--no-robust", dest="robust", action="store_false")
    ap.add_argument("--robust-c", type=float, default=3.0)
    ap.add_argument("--n-irls", type=int, default=2)
    ap.add_argument("--no-fit-bg", dest="fit_bg", action="store_false")
    ap.add_argument("--param-V", dest="param_V", action="store_true",
                    help="Включить параметрический V")
    ap.add_argument("--no-param-V", dest="param_V", action="store_false",
                    help="Отключить параметрический V (свободная V). "
                         "Это дефолт (финальный вариант).")
    ap.set_defaults(param_V=False)
    ap.add_argument("--no-refit-V", dest="refit_V", action="store_false")
    ap.add_argument("--warmup-V", type=int, default=2)
    ap.add_argument("--lambda-theta", type=float, default=1.0)
    ap.add_argument("--gamma-min", type=float, default=0.1)
    ap.add_argument("--no-sum2v", dest="sum2v", action="store_false")
    ap.add_argument("--center-offset", type=float, nargs=2, default=(0.0, 0.0))
    ap.add_argument("--center-offset-mode",
                    choices=("none", "auto", "manual", "smooth-argmax"),
                    default="smooth-argmax")
    ap.add_argument("--zero-weight-mode",
                    choices=("zero", "one"), default="one")
    ap.add_argument("--eps-mode", choices=("physical", "strict"),
                    default="strict")
    ap.add_argument("--detrend-order",
                    choices=("none", "plane", "quadratic",
                             "periphery_plane", "periphery_quadratic",
                             "high_pass"),
                    default=None,
                    help="None -- по --eps-mode: physical -> periphery_quadratic, "
                         "strict -> high_pass.")
    ap.add_argument("--center-x", type=int, nargs=2, default=(13, 16))
    ap.add_argument("--center-y", type=int, nargs=2, default=(9, 11))
    ap.add_argument("--high-pass-sigma", type=float, default=2.0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    # None -> режим по умолчанию
    if args.detrend_order is None:
        if args.eps_mode == "physical":
            args.detrend_order = "periphery_quadratic"
        elif args.eps_mode == "strict":
            args.detrend_order = "high_pass"
        else:
            args.detrend_order = "plane"
    log(f"eps-mode = {args.eps_mode}, detrend-order = {args.detrend_order}")
    log(f"high-pass-sigma = {args.high_pass_sigma}, "
        f"r-max = {args.r_max}, lam-eps = {args.lam_eps}, "
        f"n-iter = {args.n_iter}, param_V = {args.param_V}, "
        f"zero-weight-mode = {args.zero_weight_mode}, "
        f"center-offset-mode = {args.center_offset_mode}")

    PADS_X, PADS_Y = args.pads_x, args.pads_y
    CX, CY = PADS_X // 2, PADS_Y // 2

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
    # нормировка на ток: I0 = median(I)
    I0 = float(np.median(I_vals[I_vals > 0]))
    data = data_raw * (I0 / I_vals)[:, None, None]
    log(f"Read {T} frames (skipped {n_skip})")
    cx_arr = np.array([c[0] for c in centers], dtype=float)
    cy_arr = np.array([c[1] for c in centers], dtype=float)
    log(f"c_x: mean={cx_arr.mean():.2f} std={cx_arr.std():.2f} "
        f"min={cx_arr.min():.0f} max={cx_arr.max():.0f}")
    log(f"c_y: mean={cy_arr.mean():.2f} std={cy_arr.std():.2f} "
        f"min={cy_arr.min():.0f} max={cy_arr.max():.0f}")

    if args.center_offset_mode == "smooth-argmax":
        from scipy.ndimage import median_filter
        T_all = data.shape[0]
        amax_x = np.zeros(T_all, dtype=float)
        amax_y = np.zeros(T_all, dtype=float)
        peak_val = np.zeros(T_all)
        for t in range(T_all):
            idx = np.argmax(data[t])
            iy, ix = np.unravel_index(idx, (PADS_Y, PADS_X))
            amax_x[t] = ix
            amax_y[t] = iy
            peak_val[t] = data[t, iy, ix]
        margin = 2
        inside = ((cx_arr >= margin) & (cx_arr <= PADS_X - 1 - margin) &
                  (cy_arr >= margin) & (cy_arr <= PADS_Y - 1 - margin))
        n_inside = int(inside.sum())
        # "полные" кадры: центр внутри и пик не размыт
        if n_inside > 0:
            peak_med_inside = float(np.median(peak_val[inside]))
            full = inside & (peak_val > 0.3 * peak_med_inside)
        else:
            full = inside.copy()
        if int(full.sum()) < 10 and n_inside >= 10:
            full = inside.copy()
            log(f"smooth-argmax: fallback на margin-only ({n_inside} кадров)")
        n_full = int(full.sum())
        log(f"\n=== smooth-argmax: полных кадров {n_full} / {T_all} "
            f"(inside={n_inside}) ===")
        if n_full < 10:
            log("smooth-argmax: мало полных кадров, поправка не применяется")
        else:
            delta_x = np.full(T_all, np.nan)
            delta_y = np.full(T_all, np.nan)
            delta_x[full] = amax_x[full] - cx_arr[full]
            delta_y[full] = amax_y[full] - cy_arr[full]
            dx_s = median_filter(np.nan_to_num(delta_x, nan=0.0), size=11)
            dy_s = median_filter(np.nan_to_num(delta_y, nan=0.0), size=11)
            idx_full = np.where(full)[0]
            if idx_full.size >= 2:
                dx_s = np.interp(np.arange(T_all), idx_full, dx_s[idx_full])
                dy_s = np.interp(np.arange(T_all), idx_full, dy_s[idx_full])
            dx_s = np.clip(dx_s, -6.0, 6.0)
            dy_s = np.clip(dy_s, -6.0, 6.0)
            log(f"smooth-argmax: dx mean={dx_s.mean():+.3f} "
                f"std={dx_s.std():.3f} min={dx_s.min():+.2f} "
                f"max={dx_s.max():+.2f}")
            log(f"smooth-argmax: dy mean={dy_s.mean():+.3f} "
                f"std={dy_s.std():.3f} min={dy_s.min():+.2f} "
                f"max={dy_s.max():+.2f}")
            cx_arr = cx_arr + dx_s
            cy_arr = cy_arr + dy_s
            centers = [(float(cx_arr[t]), float(cy_arr[t]))
                       for t in range(T_all)]
            log("smooth-argmax: центры обновлены (c_x[t] += dx[t])")
            np.savetxt(os.path.join(args.out, "center_smooth.txt"),
                       np.c_[np.arange(T_all), cx_arr, cy_arr,
                             dx_s, dy_s, full.astype(int)],
                       header="t c_x_corr c_y_corr delta_x delta_y full")

    dx, dy = build_geometry(centers, PADS_Y, PADS_X)
    T_all = data.shape[0]
    argmax_x = np.zeros(T_all, dtype=int)
    argmax_y = np.zeros(T_all, dtype=int)
    peak_val = np.zeros(T_all)
    for t in range(T_all):
        idx = np.argmax(data[t])
        argmax_y[t], argmax_x[t] = np.unravel_index(idx, (PADS_Y, PADS_X))
        peak_val[t] = data[t, argmax_y[t], argmax_x[t]]
    margin = 2
    inside = ((cx_arr >= margin) & (cx_arr <= PADS_X - 1 - margin) &
              (cy_arr >= margin) & (cy_arr <= PADS_Y - 1 - margin))
    if inside.any():
        peak_med_inside = float(np.median(peak_val[inside]))
        full = inside & (peak_val > 0.3 * peak_med_inside)
    else:
        full = inside.copy()
    if int(full.sum()) < 10 and int(inside.sum()) >= 10:
        full = inside.copy()
    n_full = int(full.sum())
    log(f"\n=== center diagnostics: argmax(data) - (c_x, c_y) ===")
    log(f"полных кадров: {n_full} / {T_all}")
    if n_full > 0:
        shift_x_obs = (argmax_x[full] - cx_arr[full]).astype(int)
        shift_y_obs = (argmax_y[full] - cy_arr[full]).astype(int)
        log(f"dx: mean={shift_x_obs.mean():+.3f}  std={shift_x_obs.std():.3f}  "
            f"min={shift_x_obs.min():+d}  max={shift_x_obs.max():+d}")
        log(f"dy: mean={shift_y_obs.mean():+.3f}  std={shift_y_obs.std():.3f}  "
            f"min={shift_y_obs.min():+d}  max={shift_y_obs.max():+d}")
    n_zero_2d = np.sum(data_raw <= 0.0, axis=0)
    n_frames = T_all
    zero_always = (n_zero_2d == n_frames)
    zero_sometimes = (n_zero_2d > 0) & (n_zero_2d < n_frames)
    log(f"\n=== zero diagnostics ===")
    log(f"пикселей с нулём во ВСЕХ кадрах: {int(zero_always.sum())}")
    log(f"пикселей с нулём в НЕКОТОРЫХ кадрах: {int(zero_sometimes.sum())}")
    co_dx, co_dy = 0.0, 0.0
    if args.center_offset_mode == "manual":
        co_dx, co_dy = args.center_offset
    elif args.center_offset_mode == "auto":
        if n_full > 0:
            co_dx = float(np.round(np.median(shift_x_obs)))
            co_dy = float(np.round(np.median(shift_y_obs)))
    log(f"\ncenter-offset mode={args.center_offset_mode}: "
        f"применяю dx={co_dx:+.3f}, dy={co_dy:+.3f}")
    dx = dx - co_dx
    dy = dy - co_dy
    r = np.sqrt(dx ** 2 + dy ** 2)
    mask = r < args.r_max
    n_valid_2d = mask.sum(axis=0)
    pix_ok = n_valid_2d >= args.min_valid
    mask &= pix_ok[None, :, :]
    log(f"mask: n_valid center = {n_valid_2d[CY, CX]}, "
        f"valid pixels = {pix_ok.sum()} / {PADS_Y * PADS_X}")
    with open(os.path.join(args.out, "center_diag.txt"), "w") as f:
        f.write("# t c_x c_y argmax_x argmax_y dx dy full\n")
        for t in range(T_all):
            f.write(f"{t} {centers[t][0]} {centers[t][1]} "
                    f"{argmax_x[t]} {argmax_y[t]} "
                    f"{argmax_x[t] - int(centers[t][0]):+d} "
                    f"{argmax_y[t] - int(centers[t][1]):+d} "
                    f"{1 if full[t] else 0}\n")
    with open(os.path.join(args.out, "zero_diag.txt"), "w") as f:
        f.write("# x y n_zero n_frames zero_always zero_sometimes\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {int(n_zero_2d[i,j])} {n_frames} "
                        f"{1 if zero_always[i,j] else 0} "
                        f"{1 if zero_sometimes[i,j] else 0}\n")
    # dead-from-data: нуль во ВСЕХ кадрах
    dead_from_data = zero_always.copy()
    dead_thresh_eps = float(args.dead_thresh_eps)
    dead_thresh_stat = float(args.dead_thresh_stat)
    if dead_thresh_eps <= 0.0:
        log("dead: авто-детекция по eps отключена")
    dead_init = read_dead_file(args.read_dead, PADS_Y, PADS_X)
    dead_init = dead_init | dead_from_data
    log(f"dead: из данных (нуль во всех {T} кадрах) = {int(dead_from_data.sum())}; "
        f"итоговая стартовая маска = {int(dead_init.sum())}")
    log(f"\n=== fit M12 (param_V={args.param_V}, sum2v={args.sum2v}, "
        f"refit_V={args.refit_V}, warmup_V={args.warmup_V}, "
        f"lambda_theta={args.lambda_theta}, detrend={args.detrend_order}, "
        f"gamma_min={args.gamma_min}, robust={args.robust}, "
        f"fit_bg={args.fit_bg}, zero_weight_mode={args.zero_weight_mode}) ===")
    res = fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X,
                  n_iter=args.n_iter, lam_eps=args.lam_eps,
                  step=args.step, sigma_smooth=args.sigma_smooth,
                  fit_sat=args.fit_sat, n_sat_init=args.n_sat_init, log=log,
                  dead_init=dead_init, n_valid_2d=n_valid_2d,
                  dead_thresh_eps=dead_thresh_eps,
                  dead_thresh_stat=dead_thresh_stat,
                  robust=args.robust, robust_c=args.robust_c,
                  n_irls=args.n_irls,
                  fit_bg=args.fit_bg, param_V=args.param_V,
                  refit_V=args.refit_V,
                  warmup_V=args.warmup_V,
                  lambda_theta=args.lambda_theta,
                  detrend_order=args.detrend_order,
                  gamma_min=args.gamma_min, sum2v=args.sum2v,
                  zero_weight_mode=args.zero_weight_mode,
                  center_x=args.center_x, center_y=args.center_y,
                  high_pass_sigma=args.high_pass_sigma)
    eps = res["eps"]; A = res["A"]; n_sat = res["N_sat"]
    dead = res["dead"]; b = res["b"]
    V_params = res["V_params"]; param_V = res["param_V"]; sum2v = res["sum2v"]
    shift_x, shift_y = res["shift"]
    dx_c = res["dx_c"]; dy_c = res["dy_c"]; V_grid0 = res["V_grid"]
    DXg, DYg = np.meshgrid(dx_c - shift_x, dy_c - shift_y, indexing="xy")
    if param_V and V_params is not None:
        if sum2v:
            V_out = V_param2(DXg, DYg, *V_params)
        else:
            V_out = V_param(DXg, DYg, *V_params)
        V_out = V_out / V_out.max()
    else:
        V_out = V_grid0 / V_grid0.max()
    np.savetxt(os.path.join(args.out, "V_dxdy.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), V_out.ravel()],
               header="dx dy V")
    np.savetxt(os.path.join(args.out, "V_emp_dxdy.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), V_grid0.ravel()],
               header="dx dy V_emp")
    if param_V and V_params is not None:
        with open(os.path.join(args.out, "V_params.txt"), "w") as f:
            if sum2v:
                f.write("# A sx1 sy1 theta_deg eta gamma sx2 sy2 w "
                        "shift_x shift_y\n")
                f.write(f"{V_params[0]:.10e} {V_params[1]:.10e} "
                        f"{V_params[2]:.10e} {np.degrees(V_params[3]):.6f} "
                        f"{V_params[4]:.10e} {V_params[5]:.10e} "
                        f"{V_params[6]:.10e} {V_params[7]:.10e} "
                        f"{V_params[8]:.10e} "
                        f"{shift_x:.6f} {shift_y:.6f}\n")
            else:
                f.write("# A sx sy theta_deg eta gamma shift_x shift_y\n")
                f.write(f"{V_params[0]:.10e} {V_params[1]:.10e} "
                        f"{V_params[2]:.10e} {np.degrees(V_params[3]):.6f} "
                        f"{V_params[4]:.10e} {V_params[5]:.10e} "
                        f"{shift_x:.6f} {shift_y:.6f}\n")
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
    log(f"{'r':>8s} {'V_emp':>12s} {'V_model*scale':>14s} {'ratio':>10s}")
    for k in range(0, len(cb), 2):
        vm = V_r_model[k] * scale
        ratio = vm / V_emp_r[k] if V_emp_r[k] > 1e-12 else 0.0
        log(f"{cb[k]:8.2f} {V_emp_r[k]:12.4e} {vm:14.4e} {ratio:10.3f}")
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
    log(f"frac(|r|>3*rms) = {frac_3s:.3%}, "
        f"frac(|r|>5*rms) = {frac_5s:.3%}")
    log(f"frac(|r|>3*sigma_MAD) = {frac_3mad:.3%}, "
        f"frac(|r|>5*sigma_MAD) = {frac_5mad:.3%}")
    log(f"chi2_w/M = {chi2_total / D.size:.4e}")
    log(f"chi2: total = {chi2_total:.4e}, "
        f"outliers(|r|>5*rms) = {chi2_out5:.4e} "
        f"({100 * chi2_out5_frac:.1f}% от chi2)")
    log(f"Wf/Wf_base: median = {np.median(w_ratio):.3f}, "
        f"min = {np.min(w_ratio):.3e}")
    log(f"b: mean = {b.mean():.4e}, max = {b.max():.4e}, "
        f"min = {b.min():.4e}, median = {np.median(b):.4e}")
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
    flat_order = np.argsort(np.nan_to_num(mean_abs, nan=-1).ravel())[::-1]
    log("\n=== топ-10 пикселей по mean|resid| ===")
    log(f"{'x':>4s} {'y':>4s} {'mean|r|':>12s} {'mean_r':>12s} "
        f"{'n':>6s} {'eps':>10s} {'b':>10s} {'dead':>4s}")
    for k in flat_order[:10]:
        i, j = divmod(int(k), PADS_X)
        log(f"{j:4d} {i:4d} {mean_abs[i,j]:12.4e} {mean_resid[i,j]:12.4e} "
            f"{cnt_2d[i,j]:6d} {eps[i,j]:10.3e} {b[i,j]:10.3e} "
            f"{1 if dead[i,j] else 0:4d}")
    if args.kfold > 1:
        log(f"\n=== sigma_stat (kfold K={args.kfold}) ===")
        eps_stat, folds = kfold_stat(
            res, PADS_Y, PADS_X, lam_eps=args.lam_eps,
            K=args.kfold, detrend_order=args.detrend_order, log=log,
            center_x=args.center_x, center_y=args.center_y,
            high_pass_sigma=args.high_pass_sigma)
    else:
        log("\n=== sigma_stat (split-half) ===")
        eps_stat, folds = kfold_stat(
            res, PADS_Y, PADS_X, lam_eps=args.lam_eps, K=2,
            detrend_order=args.detrend_order, log=log,
            center_x=args.center_x, center_y=args.center_y,
            high_pass_sigma=args.high_pass_sigma)
    r_2d = np.sqrt(
        (np.arange(PADS_X, dtype=float)[None, :] - CX) ** 2 +
        (np.arange(PADS_Y, dtype=float)[:, None] - CY) ** 2
    )
    eps_stat[r_2d > args.r_sigma_max] = np.nan
    log(f"sigma_stat = nan для r > {args.r_sigma_max}: "
        f"{(r_2d > args.r_sigma_max).sum()} пикселей")
    dead_final = dead | detect_dead(eps, eps_stat, n_valid_2d,
                                    PADS_Y, PADS_X,
                                    thresh_eps=dead_thresh_eps,
                                    thresh_stat=dead_thresh_stat)
    if int(dead_final.sum()) != int(dead.sum()):
        log(f"dead: уточнено по sigma_stat: {int(dead.sum())} -> "
            f"{int(dead_final.sum())}")
        eps[dead_final] = 0.0
        b[dead_final] = 0.0
        eps_stat[dead_final] = np.nan
        dead = dead_final
    if args.real_eps and os.path.exists(args.real_eps):
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
        log(f"[compare] chi2 = {chi2:.2f}, n_good = {n_good}, "
            f"chi2/n = {chi2/n_good:.3f}, scale = {scale_f:.3f}")
    else:
        log("\n(real_eps не задан или не найден)")
        eps_sys = np.full_like(eps_stat, np.nan)
        eps_tot = eps_stat.copy()
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
    log(f"\n=== beam sigma (по P_avg, с вычетом пьедестала) ===")
    log(f"x: sigma={sx_m:.4f}px  mean={mx:.3f}  bg={bgx:.3e}")
    log(f"y: sigma={sy_m:.4f}px  mean={my:.3f}  bg={bgy:.3e}")
    if args.pad_mm:
        log(f"x: sigma={sx_m * args.pad_mm[0]:.4f}mm")
        log(f"y: sigma={sy_m * args.pad_mm[1]:.4f}mm")
    if args.fit_sat:
        log(f"\nN_sat = {n_sat:.3e}")
    if param_V and V_params is not None:
        log(f"\n=== V параметры ({'сумма 2 Voigt' if sum2v else 'одиночный Voigt'}) ===")
        log(f"A = {V_params[0]:.6e}")
        log(f"sigma_x1 = {V_params[1]:.4f} px")
        log(f"sigma_y1 = {V_params[2]:.4f} px")
        log(f"theta = {np.degrees(V_params[3]):.3f}deg")
        log(f"eta (доля Коши) = {V_params[4]:.4f}")
        log(f"gamma (ширина Коши) = {V_params[5]:.4f}")
        if sum2v:
            log(f"sigma_x2 = {V_params[6]:.4f} px (широкий Гаусс)")
            log(f"sigma_y2 = {V_params[7]:.4f} px (широкий Гаусс)")
            log(f"w = {V_params[8]:.4f} (вес Voigt)")
        log(f"центр: dx = {shift_x:.3f}, dy = {shift_y:.3f}")
    with open(os.path.join(out, "fit_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
