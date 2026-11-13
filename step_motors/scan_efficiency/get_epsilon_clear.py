#!/usr/bin/env python3
"""
get_epsilon_clear.py -- восстановление карты эффективности пикселей eps
по кадрам со сканирующим пучком (упрощённая версия, только strict).

Модель:
    data[t,i,j] ~= eps[i,j] * A_t * V(dx, dy) + b[i,j]
    dx = j - c_x[t] - center_offset_x
    dy = i - c_y[t] - center_offset_y

V(dx,dy) -- свободная эмпирическая: биннинг data по (dx,dy),
заливка "дыр", сглаживание, интерполяция. Параметрическая V не
используется (ср. с полной get_epsilon.py, где есть --param-V).

Режим: strict (единственный).
    high_pass sigma=2 убирает всю крупномасштабную структуру
    (градиенты, виньетирование), оставляя локальные дефекты < 2 px:
        eps_hp = eps - gaussian_filter(eps, 2.0) + mean(eps).
    Это финальный вариант: corr ~ 0.56, rms ~ 0.05 на двух наборах
    данных (в пределах r <= 13 от центра).

Запуск:
    python get_epsilon_clear.py --data <file> --out <dir>

    Дефолты соответствуют финальному варианту:
      --high-pass-sigma 2.0
      --lam-eps 0.01
      --r-max 13
      --n-iter 15
      --center-offset-mode smooth-argmax
      --drop-zeros (True = нули -- выпадения)

Ключевые решения:
  * data нормируется на ток I (I0 = median(I)).
  * Closed-form eps, A_t, b (альтернирующий МНК).
  * Тихонов по eps через лапласиан (lam_eps=0.01).
  * Робастные веса (Huber) с защитой от MAD=0, n_irls=2.
  * sigma_stat по K=4 фолдам (std * sqrt(K-1)).
  * Мёртвые пиксели:
      - dead-from-data: нуль во ВСЕХ кадрах -> dead (всегда);
      - авто-детекция по eps: только если --dead-thresh-eps > 0.
  * W_base = 1/max(data, 1); в режиме --keep-zeros нули -- сигнал,
    иначе (по умолчанию) вес 0 -- выпадения.
  * Аддитивный фон b[i,j].
  * Диагностика: center_diag, zero_diag, center_smooth, остатки,
    карты, топ-10.

Выход: epsilon.txt, avg_data.txt/.npy, V_dxdy.txt, V_emp_dxdy.txt,
       V_r.txt, V_dxdy_amp.txt, bkg.npy, bkg.txt, P_t.npy, eps.npy,
       A.npy, eps_folds.npy, fit_log.txt, resid_mean.npy,
       resid_abs_mean.npy, resid_counts.npy, resid_map.txt,
       center_diag.txt, zero_diag.txt
       (+ center_smooth.txt при smooth-argmax, + маска если --save-dead).
"""

import os
import time
import argparse
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import gaussian_filter, median_filter
from scipy.interpolate import RegularGridInterpolator


#                            constants
_DEAD_PENALTY = 1e6            # штраф диагонали для dead-пикселей
_DIAG_EPS = 1e-9               # добавка к диагонали при вырождении
_SMOOTH_ARGMAX_CLIP = 6.0      # клип delta_x/delta_y в smooth-argmax
_SMOOTH_ARGMAX_MARGIN = 2      # отступ от края для "полных" кадров
_SMOOTH_ARGMAX_PEAK_FRAC = 0.3 # порог пика для "полных" кадров
_ROBUST_W_MIN = 0.01           # нижняя граница робастного веса
_R_SIGMA_MAX = 14.0            # sigma_stat = nan при r > этого
_HIGH_PASS_SIGMA = 2.0
_LAM_EPS = 0.01
_R_MAX = 13.0
_MIN_VALID = 200
_N_ITER = 15


#                            logger
class Logger:
    """Печатает в stdout и копит строки для fit_log.txt."""

    def __init__(self, echo: bool = True):
        self.echo = echo
        self.lines: list[str] = []

    def __call__(self, msg: str):
        if self.echo:
            print(msg)
        self.lines.append(str(msg))

    def save(self, path: str):
        with open(path, "w") as f:
            f.write("\n".join(self.lines))


#                            config
@dataclass
class Config:
    """Все параметры фита в одном месте. Дефолты -- strict."""

    data: str = "data/data20261109_143025.txt"
    out: str = "."
    pads_x: int = 32
    pads_y: int = 20
    # маска
    r_max: float = _R_MAX
    min_valid: int = _MIN_VALID
    # фит
    n_iter: int = _N_ITER
    lam_eps: float = _LAM_EPS
    step: float = 1.0
    sigma_smooth: float = 0.5
    kfold: int = 4
    # robust
    robust: bool = True
    robust_c: float = 3.0
    n_irls: int = 2
    # веса: True = нули -- выпадения (вес 0), False = нули -- сигнал (вес 1)
    drop_zeros: bool = True
    # detrend
    high_pass_sigma: float = _HIGH_PASS_SIGMA
    # центрирование
    center_offset_mode: str = "smooth-argmax"
    # dead
    read_dead: Optional[str] = None
    save_dead: Optional[str] = None
    dead_thresh_eps: float = 0.05
    dead_thresh_stat: float = 0.01


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


#                            empirical V
def empirical_V_2d(dx, dy, data, mask, step=1.0, sigma_smooth=1.0):
    """V_emp: биннинг data по (dx,dy) + заливка дыр + сглаживание.

    Возвращает (dx_c, dy_c, V_norm, counts); V_norm нормирована на max.
    """
    t_idx, i_idx, j_idx = np.nonzero(mask)
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]
    D = data[t_idx, i_idx, j_idx]
    dx_c = np.arange(np.floor(DX.min()), np.ceil(DX.max()) + 1, step)
    dy_c = np.arange(np.floor(DY.min()), np.ceil(DY.max()) + 1, step)
    ix = np.clip(np.rint((DX - dx_c[0]) / step).astype(int), 0, len(dx_c) - 1)
    iy = np.clip(np.rint((DY - dy_c[0]) / step).astype(int), 0, len(dy_c) - 1)
    H = np.zeros((len(dy_c), len(dx_c)))
    C = np.zeros((len(dy_c), len(dx_c)))
    np.add.at(H, (iy, ix), D)
    np.add.at(C, (iy, ix), 1.0)
    # среднее по бину; пустые бины -> 0, затем сглаживаются mode='nearest'
    V = np.where(C > 0, H / np.maximum(C, 1e-9), 0.0)
    V = gaussian_filter(V, sigma=sigma_smooth, mode="nearest")
    Vmax = V.max()
    if Vmax <= 0:
        raise ValueError("V_emp is zero everywhere")
    return dx_c, dy_c, V / Vmax, C


#                            sparse Laplacian (Neumann)
def build_laplacian_sparse(Y, X):
    """5-точечный лапласиан YxX в разреженном виде. Границы -- Neumann."""
    n = Y * X
    main = np.full(n, 4.0)
    # углы -- 2, рёбра -- 3
    main[0] = main[X - 1] = main[n - X] = main[n - 1] = 2.0
    main[1:X - 1] = main[n - X + 1:n - 1] = 3.0
    main[np.arange(X, n - X, X)] = 3.0
    main[np.arange(2 * X - 1, n - X, X)] = 3.0
    # off_x -- соседи по строке, off_y -- между строками
    off_x = -np.ones(n - 1); off_x[X - 1::X] = 0.0
    off_y = -np.ones(n - X)
    return (diags([main, off_x, off_x, off_y, off_y],
                  [0, 1, -1, X, -X], shape=(n, n)).tocsr())


#                            dead pixels
def detect_dead(eps, eps_stat, thresh_eps, thresh_stat):
    """dead = (eps < thresh_eps) & (eps_stat < thresh_stat), если пороги > 0."""
    if thresh_eps <= 0.0:
        return np.zeros_like(eps, dtype=bool)
    dead = np.isfinite(eps) & (eps < thresh_eps)
    if eps_stat is not None:
        dead &= np.isfinite(eps_stat) & (eps_stat < thresh_stat)
    return dead


def read_dead_file(path, PADS_Y, PADS_X):
    """Формат: x y dead_flag (1=dead, 0=alive), по строке на пиксель."""
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


#                            closed-form eps, A_t, b
def solve_A(D_lin, W_lin, ev, lin_A, T):
    """A_t = sum(w*D*e) / sum(w*e^2) по кадрам (нижняя граница 1e-6)."""
    num = np.bincount(lin_A, weights=W_lin * D_lin * ev, minlength=T)
    den = np.bincount(lin_A, weights=W_lin * ev * ev, minlength=T)
    return np.maximum(
        np.where(den > 0, num / np.maximum(den, 1e-300), 1e-6), 1e-6)


def solve_eps(D_lin, W_lin, av, lin_eps, N_eps, L, lam_eps,
              PADS_Y, PADS_X, dead=None):
    """eps = argmin sum w(D - eps*A*V)^2 + lam_eps*diag_mean*||L*eps||^2.

    dead-пиксели -- большой штраф на диагональ.
    """
    rhs = np.bincount(lin_eps, weights=W_lin * D_lin * av, minlength=N_eps)
    diag = np.bincount(lin_eps, weights=W_lin * av * av, minlength=N_eps)
    dead_flat = dead.ravel() if dead is not None and dead.any() else None

    if lam_eps > 0 and L is not None:
        diag_mean = diag[~dead_flat].mean() if dead_flat is not None \
            else diag.mean()
        if not np.isfinite(diag_mean) or diag_mean <= 0:
            diag_mean = 1.0
        if dead_flat is not None:
            rhs[dead_flat] = 0.0
            diag = diag + np.where(dead_flat,
                                   _DEAD_PENALTY * max(diag_mean, 1.0), 0.0)
        # M = diag + lam*diag_mean*L^T*L (+ малая добавка против сингулярности)
        M = (diags(diag, 0) + lam_eps * diag_mean * (L.T @ L)
             + diags(np.full(N_eps, _DIAG_EPS * max(diag_mean, 1.0)), 0))
        eps = spsolve(M.tocsc(), rhs)
        if not np.all(np.isfinite(eps)):
            eps = np.where(diag > 0, rhs / np.maximum(diag, 1e-300), 1.0)
    else:
        eps = np.where(diag > 0, rhs / np.maximum(diag, 1e-300), 1.0)

    eps = np.maximum(eps, 0.0)
    if dead_flat is not None:
        eps[dead_flat] = 0.0
    return eps.reshape(PADS_Y, PADS_X).ravel()


def solve_b(D_lin, W_lin, model_no_b, lin_eps, N_eps, dead=None):
    """b = средний остаток (D - eps*A*V) по точкам пикселя."""
    num = np.bincount(lin_eps, weights=W_lin * (D_lin - model_no_b),
                      minlength=N_eps)
    den = np.bincount(lin_eps, weights=W_lin, minlength=N_eps)
    b = np.where(den > 0, num / np.maximum(den, 1e-300), 0.0)
    if dead is not None and dead.any():
        b[dead.ravel()] = 0.0
    return b


#                            robust weights (Huber)
def robust_weights(resid, c=3.0):
    med = np.median(resid)
    mad = np.median(np.abs(resid - med))
    sigma = 1.4826 * mad
    # защита от MAD=0 (все остатки одинаковые) -- иначе деление на 0
    if not np.isfinite(sigma) or sigma <= 1e-12:
        return np.ones_like(resid)
    r = np.abs(resid - med) / sigma
    return np.maximum(np.where(r <= c, 1.0, c / np.maximum(r, 1e-12)),
                      _ROBUST_W_MIN)


#                            detrend: только high_pass
def high_pass_eps(eps_2d, sigma=_HIGH_PASS_SIGMA, dead=None):
    """eps_hp = eps - gaussian(eps, sigma) + mean(eps), clip >= 0."""
    sm = gaussian_filter(eps_2d, sigma=sigma)
    out = np.maximum(eps_2d - sm + sm.mean(), 0.0)
    if dead is not None and dead.any():
        out[dead] = 0.0
    return out


#                            one eps/A/b fit step
def _step(D, W, V_tab, *, eps, b, A,
          lin_eps, lin_A, T, N_eps,
          L, lam_eps, PADS_Y, PADS_X, dead,
          high_pass_sigma, fit_bg):
    D_lin = D - b[lin_eps]
    A = solve_A(D_lin, W, eps[lin_eps] * V_tab, lin_A, T)
    eps = solve_eps(D_lin, W, A[lin_A] * V_tab,
                    lin_eps, N_eps, L, lam_eps, PADS_Y, PADS_X, dead=dead)
    if not np.all(np.isfinite(eps)):
        return None
    eps_2d = high_pass_eps(eps.reshape(PADS_Y, PADS_X),
                           sigma=high_pass_sigma, dead=dead).ravel()
    alive = ~dead.ravel() if dead is not None and (~dead).any() \
        else np.ones(N_eps, bool)
    m = max(eps_2d[alive].mean(), 1e-12)
    eps = eps_2d / m
    if dead is not None:
        eps[dead.ravel()] = 0.0
    A = solve_A(D - b[lin_eps], W, eps[lin_eps] * V_tab, lin_A, T) * m
    if fit_bg:
        b = solve_b(D, W, eps[lin_eps] * A[lin_A] * V_tab,
                    lin_eps, N_eps, dead=dead)
    return eps.reshape(PADS_Y, PADS_X), A, b.reshape(PADS_Y, PADS_X)


#                            main fit
def fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X, cfg: Config,
            log: Logger, dead_init=None):
    T = data.shape[0]
    N_eps = PADS_Y * PADS_X

    # W_base: 1/max(data,1); при drop_zeros нули -> вес 0
    if cfg.drop_zeros:
        W_base = np.where(data_raw > 0.0, 1.0 / np.maximum(data_raw, 1.0), 0.0)
    else:
        W_base = 1.0 / np.maximum(data_raw, 1.0)

    t_idx, i_idx, j_idx = np.nonzero(mask)
    D = data[t_idx, i_idx, j_idx]
    Wf_base = W_base[t_idx, i_idx, j_idx]
    DX = dx[t_idx, i_idx, j_idx]
    DY = dy[t_idx, i_idx, j_idx]
    lin_eps = i_idx * PADS_X + j_idx
    lin_A = t_idx
    log(f"fit points: M = {D.size} / {T * PADS_Y * PADS_X} "
        f"({100.0 * D.size / (T * PADS_Y * PADS_X):.1f}%)")

    dx_c, dy_c, V_grid0, C_grid = empirical_V_2d(
        dx, dy, data, mask, step=cfg.step, sigma_smooth=cfg.sigma_smooth)
    log(f"V_emp 2D (init): shape={V_grid0.shape}, "
        f"bins with data: {(C_grid > 0).sum()} / {C_grid.size}")
    # сдвиг профиля V: центр = максимум эмпирической V
    iy_max, ix_max = np.unravel_index(np.argmax(V_grid0), V_grid0.shape)
    shift_x, shift_y = dx_c[ix_max], dy_c[iy_max]
    log(f"V center offset: dx={shift_x:.2f}, dy={shift_y:.2f}")
    dx = dx - shift_x; dy = dy - shift_y
    DX = dx[t_idx, i_idx, j_idx]; DY = dy[t_idx, i_idx, j_idx]

    interp = RegularGridInterpolator((dy_c, dx_c), V_grid0,
                                     bounds_error=False, fill_value=0.0)
    V_tab = interp(np.column_stack([DY + shift_y, DX + shift_x]))
    log(f"V_tab: max={V_tab.max():.3f}, mean={V_tab.mean():.3f}")

    L = build_laplacian_sparse(PADS_Y, PADS_X) if cfg.lam_eps > 0 else None
    dead = dead_init.copy() if dead_init is not None else \
        np.zeros((PADS_Y, PADS_X), dtype=bool)
    log(f"dead: стартовая маска = {int(dead.sum())} пикселей")

    eps = np.ones(N_eps); b = np.zeros(N_eps)
    if dead.any():
        eps[dead.ravel()] = 0.0; b[dead.ravel()] = 0.0
    Wf = Wf_base.copy()
    A = solve_A(D, Wf, eps[lin_eps] * V_tab, lin_A, T)
    prev = None; t0 = time.time()

    for it in range(cfg.n_iter):
        model = None
        for irls in range(cfg.n_irls if cfg.robust else 1):
            out = _step(D, Wf, V_tab,
                        eps=eps, b=b, A=A,
                        lin_eps=lin_eps, lin_A=lin_A, T=T, N_eps=N_eps,
                        L=L, lam_eps=cfg.lam_eps,
                        PADS_Y=PADS_Y, PADS_X=PADS_X, dead=dead,
                        high_pass_sigma=cfg.high_pass_sigma,
                        fit_bg=True)
            if out is None:
                log(f"  it {it:2d}.{irls}: eps содержит nan/inf -- прерываем")
                break
            eps_2d, A, b_2d = out
            eps = eps_2d.ravel(); b = b_2d.ravel()
            model = eps[lin_eps] * A[lin_A] * V_tab + b[lin_eps]
            if cfg.robust and irls < cfg.n_irls - 1:
                Wf = Wf_base * robust_weights(D - model, c=cfg.robust_c)
        if model is None:
            break
        loss_v = float(np.sum(Wf * (D - model) ** 2))
        d_loss = 0.0 if prev is None else abs(loss_v - prev) / max(abs(prev), 1e-30)
        log(f"  it {it:2d}: loss={loss_v:.4e} d_loss={d_loss:.2e} "
            f"dead={int(dead.sum())} b_mean={b.mean():.3e} "
            f"b_min={b.min():.3e} time={time.time() - t0:.2f}s")
        if not np.isfinite(loss_v):
            log(f"  it {it:2d}: loss=nan -- прерываем"); break

        new_dead = detect_dead(eps.reshape(PADS_Y, PADS_X), None,
                               cfg.dead_thresh_eps, cfg.dead_thresh_stat)
        added = int((new_dead & ~dead).sum())
        if added:
            dead = dead | new_dead
            eps[dead.ravel()] = 0.0; b[dead.ravel()] = 0.0
            log(f"  it {it:2d}: dead += {added} -> {int(dead.sum())}")

        if prev is not None and d_loss < 1e-6:
            log(f"converged at it {it} (d_loss={d_loss:.2e})"); break
        prev = loss_v
    log(f"fit time: {time.time() - t0:.2f}s")

    out = _step(D, Wf, V_tab,
                eps=eps, b=b, A=A,
                lin_eps=lin_eps, lin_A=lin_A, T=T, N_eps=N_eps,
                L=L, lam_eps=cfg.lam_eps, PADS_Y=PADS_Y, PADS_X=PADS_X,
                dead=dead, high_pass_sigma=cfg.high_pass_sigma, fit_bg=True)
    if out is not None:
        eps_2d, A, b_2d = out
    else:
        eps_2d = eps.reshape(PADS_Y, PADS_X); b_2d = b.reshape(PADS_Y, PADS_X)
    eps = eps_2d.ravel(); b = b_2d.ravel()

    # P_t: профиль пучка на каждом кадре, без eps и b
    P_t = np.zeros((T, PADS_Y, PADS_X))
    for t in range(T):
        P_t[t] = A[t] * interp(np.column_stack(
            [(dy[t] - shift_y).ravel(), (dx[t] - shift_x).ravel()])
        ).reshape(PADS_Y, PADS_X)

    model = eps[lin_eps] * A[lin_A] * V_tab + b[lin_eps]
    resid = D - model
    return {"eps": eps_2d, "A": A, "P_t": P_t, "b": b_2d,
            "shift": (shift_x, shift_y),
            "V_tab": V_tab, "dx_c": dx_c, "dy_c": dy_c, "V_grid": V_grid0,
            "mask": mask, "dead": dead,
            "Wf": Wf, "Wf_base": Wf_base,
            "resid": resid, "resid_abs": np.abs(resid),
            "t_idx": t_idx, "i_idx": i_idx, "j_idx": j_idx,
            "D": D, "lin_eps": lin_eps, "lin_A": lin_A}


#                            sigma_stat via K folds
def kfold_stat(res, PADS_Y, PADS_X, cfg: Config, log: Logger):
    K = cfg.kfold
    T = int(res["lin_A"].max()) + 1
    N_eps = PADS_Y * PADS_X
    D = res["D"]; V_tab = res["V_tab"]; Wf = res["Wf"]
    lin_eps = res["lin_eps"]; lin_A = res["lin_A"]
    dead = res.get("dead")
    b_init = res["b"].ravel()
    L = build_laplacian_sparse(PADS_Y, PADS_X) if cfg.lam_eps > 0 else None
    folds = np.full((K, PADS_Y, PADS_X), np.nan)
    for k in range(K):
        sel = (lin_A % K) == k
        eps = np.ones(N_eps); b = b_init.copy()
        if dead is not None:
            eps[dead.ravel()] = 0.0; b[dead.ravel()] = 0.0
        A = solve_A(D[sel], Wf[sel], eps[lin_eps[sel]] * V_tab[sel],
                    lin_A[sel], T)
        for _ in range(3):
            eps = solve_eps(D[sel] - b[lin_eps[sel]], Wf[sel],
                            A[lin_A[sel]] * V_tab[sel],
                            lin_eps[sel], N_eps, L, cfg.lam_eps,
                            PADS_Y, PADS_X, dead=dead)
            if not np.all(np.isfinite(eps)):
                break
            eps = high_pass_eps(eps.reshape(PADS_Y, PADS_X),
                                sigma=cfg.high_pass_sigma, dead=dead).ravel()
            alive = ~dead.ravel() if dead is not None and (~dead).any() \
                else np.ones(N_eps, bool)
            m = max(eps[alive].mean(), 1e-12)
            eps /= m; A *= m
            if dead is not None:
                eps[dead.ravel()] = 0.0
            b = solve_b(D[sel], Wf[sel],
                        eps[lin_eps[sel]] * A[lin_A[sel]] * V_tab[sel],
                        lin_eps[sel], N_eps, dead=dead)
        eps_k = eps.reshape(PADS_Y, PADS_X)
        if np.all(np.isfinite(eps_k)):
            folds[k] = eps_k
        else:
            log(f"  kfold {k}: WARNING: нефинитный eps")
    # sigma_stat = std(folds) * sqrt(K-1) -- поправка на K фолдов
    eps_stat = np.nanstd(folds, axis=0, ddof=1) * np.sqrt(K - 1)
    log(f"kfold({K}): median sigma_stat = {np.nanmedian(eps_stat):.3e} "
        f"(нефинитных: {int(np.sum(~np.isfinite(eps_stat)))})")
    return eps_stat, folds


#                            read frames
def read_frames(path, PADS_X, PADS_Y):
    frames, centers, I_vals = [], [], []
    n_skip = 0
    need = 7 + PADS_X * PADS_Y
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            p = ln.split()
            if len(p) != need:
                n_skip += 1; continue
            try:
                I = float(p[4]); c_x = int(p[5]); c_y = int(p[6])
            except ValueError:
                n_skip += 1; continue
            if I <= 0 or not (0 <= c_x <= PADS_X and 0 <= c_y <= PADS_Y):
                n_skip += 1; continue
            fr = np.array(p[7:], dtype=float)
            if fr.size != PADS_X * PADS_Y or not np.all(np.isfinite(fr)):
                n_skip += 1; continue
            frames.append(fr); I_vals.append(I); centers.append((c_x, c_y))
    if not frames:
        raise ValueError("No valid frames")
    return (np.array(frames).reshape(-1, PADS_Y, PADS_X),
            np.array(I_vals), centers, n_skip)


#                            smooth-argmax centre correction
def smooth_argmax_correction(data, cx_arr, cy_arr, PADS_X, PADS_Y, out,
                             log: Logger):
    T = data.shape[0]
    amax_x = np.zeros(T); amax_y = np.zeros(T); peak_val = np.zeros(T)
    for t in range(T):
        iy, ix = np.unravel_index(np.argmax(data[t]), (PADS_Y, PADS_X))
        amax_x[t], amax_y[t], peak_val[t] = ix, iy, data[t, iy, ix]
    margin = _SMOOTH_ARGMAX_MARGIN
    inside = ((cx_arr >= margin) & (cx_arr <= PADS_X - 1 - margin) &
              (cy_arr >= margin) & (cy_arr <= PADS_Y - 1 - margin))
    # "полные" кадры: центр внутри и пик не размыт
    if inside.any():
        pk = float(np.median(peak_val[inside]))
        full = inside & (peak_val > _SMOOTH_ARGMAX_PEAK_FRAC * pk)
    else:
        full = inside.copy()
    if int(full.sum()) < 10 and int(inside.sum()) >= 10:
        full = inside.copy()
        log(f"smooth-argmax: fallback на margin-only ({int(inside.sum())} кадров)")
    n_full = int(full.sum())
    log(f"\n=== smooth-argmax: полных кадров {n_full} / {T} ===")
    if n_full < 10:
        log("smooth-argmax: мало полных кадров, поправка не применяется")
        return cx_arr, cy_arr, amax_x, amax_y, full
    dx_d = np.full(T, np.nan); dy_d = np.full(T, np.nan)
    dx_d[full] = amax_x[full] - cx_arr[full]
    dy_d[full] = amax_y[full] - cy_arr[full]
    dx_s = median_filter(np.nan_to_num(dx_d, nan=0.0), size=11)
    dy_s = median_filter(np.nan_to_num(dy_d, nan=0.0), size=11)
    idx_full = np.where(full)[0]
    if idx_full.size >= 2:
        grid = np.arange(T)
        dx_s = np.interp(grid, idx_full, dx_s[idx_full])
        dy_s = np.interp(grid, idx_full, dy_s[idx_full])
    dx_s = np.clip(dx_s, -_SMOOTH_ARGMAX_CLIP, _SMOOTH_ARGMAX_CLIP)
    dy_s = np.clip(dy_s, -_SMOOTH_ARGMAX_CLIP, _SMOOTH_ARGMAX_CLIP)
    log(f"smooth-argmax: dx mean={dx_s.mean():+.3f} std={dx_s.std():.3f}")
    log(f"smooth-argmax: dy mean={dy_s.mean():+.3f} std={dy_s.std():.3f}")
    np.savetxt(os.path.join(out, "center_smooth.txt"),
               np.c_[np.arange(T), cx_arr + dx_s, cy_arr + dy_s,
                     dx_s, dy_s, full.astype(int)],
               header="t c_x_corr c_y_corr delta_x delta_y full")
    return cx_arr + dx_s, cy_arr + dy_s, amax_x, amax_y, full


#                            CLI -> Config
def parse_args() -> Config:
    ap = argparse.ArgumentParser(
        description="Восстановление карты eps (strict, high_pass sigma=2, "
                    "r_max=13).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    d = Config()
    ap.add_argument("--data", default=d.data)
    ap.add_argument("--out", default=d.out)
    ap.add_argument("--pads-x", type=int, default=d.pads_x)
    ap.add_argument("--pads-y", type=int, default=d.pads_y)
    ap.add_argument("--r-max", type=float, default=d.r_max)
    ap.add_argument("--min-valid", type=int, default=d.min_valid)
    ap.add_argument("--n-iter", type=int, default=d.n_iter)
    ap.add_argument("--lam-eps", type=float, default=d.lam_eps)
    ap.add_argument("--step", type=float, default=d.step)
    ap.add_argument("--sigma-smooth", type=float, default=d.sigma_smooth)
    ap.add_argument("--kfold", type=int, default=d.kfold)
    ap.add_argument("--no-robust", dest="robust", action="store_false",
                    default=d.robust)
    ap.add_argument("--robust-c", type=float, default=d.robust_c)
    ap.add_argument("--n-irls", type=int, default=d.n_irls)
    ap.add_argument("--keep-zeros", dest="drop_zeros", action="store_false",
                    default=d.drop_zeros,
                    help="Нули -- сигнал (Poisson). По умолчанию нули "
                         "отбрасываются.")
    ap.add_argument("--high-pass-sigma", type=float, default=d.high_pass_sigma)
    ap.add_argument("--center-offset-mode",
                    choices=("none", "auto", "smooth-argmax"),
                    default=d.center_offset_mode)
    ap.add_argument("--read-dead", default=d.read_dead)
    ap.add_argument("--save-dead", default=d.save_dead)
    ap.add_argument("--dead-thresh-eps", type=float, default=d.dead_thresh_eps)
    ap.add_argument("--dead-thresh-stat", type=float,
                    default=d.dead_thresh_stat)
    return Config(**vars(ap.parse_args()))


#                            main
def main():
    cfg = parse_args()
    os.makedirs(cfg.out, exist_ok=True)
    log = Logger()
    log(f"strict: high-pass-sigma={cfg.high_pass_sigma}, r-max={cfg.r_max}, "
        f"lam-eps={cfg.lam_eps}, n-iter={cfg.n_iter}, "
        f"drop_zeros={cfg.drop_zeros}, "
        f"center-offset-mode={cfg.center_offset_mode}")

    PADS_X, PADS_Y = cfg.pads_x, cfg.pads_y
    CX, CY = PADS_X // 2, PADS_Y // 2

    data_raw, I_vals, centers, n_skip = read_frames(cfg.data, PADS_X, PADS_Y)
    T = data_raw.shape[0]
    # нормировка на ток: I0 = median(I)
    I0 = float(np.median(I_vals[I_vals > 0]))
    data = data_raw * (I0 / I_vals)[:, None, None]
    log(f"Read {T} frames (skipped {n_skip})")

    cx_arr = np.array([c[0] for c in centers], dtype=float)
    cy_arr = np.array([c[1] for c in centers], dtype=float)
    log(f"c_x: mean={cx_arr.mean():.2f} std={cx_arr.std():.2f}")
    log(f"c_y: mean={cy_arr.mean():.2f} std={cy_arr.std():.2f}")

    amax_x = amax_y = full = None
    if cfg.center_offset_mode == "smooth-argmax":
        cx_arr, cy_arr, amax_x, amax_y, full = smooth_argmax_correction(
            data, cx_arr, cy_arr, PADS_X, PADS_Y, cfg.out, log)
        centers = [(float(cx_arr[t]), float(cy_arr[t])) for t in range(T)]

    dx, dy = build_geometry(centers, PADS_Y, PADS_X)

    # если smooth-argmax не запускался -- считаем amax/full для диагностики
    if amax_x is None:
        amax_x = np.zeros(T, dtype=int); amax_y = np.zeros(T, dtype=int)
        peak_val = np.zeros(T)
        for t in range(T):
            iy, ix = np.unravel_index(np.argmax(data[t]), (PADS_Y, PADS_X))
            amax_x[t], amax_y[t], peak_val[t] = ix, iy, data[t, iy, ix]
        margin = _SMOOTH_ARGMAX_MARGIN
        inside = ((cx_arr >= margin) & (cx_arr <= PADS_X - 1 - margin) &
                  (cy_arr >= margin) & (cy_arr <= PADS_Y - 1 - margin))
        full = inside & (peak_val > _SMOOTH_ARGMAX_PEAK_FRAC *
                         float(np.median(peak_val[inside]))) if inside.any() \
            else inside.copy()
    n_full = int(full.sum())
    log(f"\n=== center diagnostics ===")
    log(f"полных кадров: {n_full} / {T}")
    if n_full > 0:
        sx = (amax_x[full] - cx_arr[full]).astype(int)
        sy = (amax_y[full] - cy_arr[full]).astype(int)
        log(f"dx: mean={sx.mean():+.3f} std={sx.std():.3f}")
        log(f"dy: mean={sy.mean():+.3f} std={sy.std():.3f}")

    n_zero_2d = np.sum(data_raw <= 0.0, axis=0)
    zero_always = (n_zero_2d == T)
    zero_sometimes = (n_zero_2d > 0) & (n_zero_2d < T)
    log(f"\n=== zero diagnostics ===")
    log(f"нулей во ВСЕХ кадрах: {int(zero_always.sum())}, "
        f"в НЕКОТОРЫХ: {int(zero_sometimes.sum())}")

    # центр. смещение: none/auto (для strict обычно 0)
    if cfg.center_offset_mode == "auto" and n_full > 0:
        sx = (amax_x[full] - cx_arr[full]).astype(int)
        sy = (amax_y[full] - cy_arr[full]).astype(int)
        co_dx = float(np.round(np.median(sx)))
        co_dy = float(np.round(np.median(sy)))
    else:
        co_dx = co_dy = 0.0
    log(f"\ncenter-offset mode={cfg.center_offset_mode}: "
        f"dx={co_dx:+.3f}, dy={co_dy:+.3f}")
    dx = dx - co_dx; dy = dy - co_dy

    r = np.sqrt(dx ** 2 + dy ** 2)
    mask = r < cfg.r_max
    n_valid_2d = mask.sum(axis=0)
    pix_ok = n_valid_2d >= cfg.min_valid
    mask &= pix_ok[None, :, :]
    log(f"mask: n_valid center = {n_valid_2d[CY, CX]}, "
        f"valid pixels = {pix_ok.sum()} / {PADS_Y * PADS_X}")

    with open(os.path.join(cfg.out, "center_diag.txt"), "w") as f:
        f.write("# t c_x c_y argmax_x argmax_y dx dy full\n")
        for t in range(T):
            f.write(f"{t} {centers[t][0]} {centers[t][1]} "
                    f"{amax_x[t]} {amax_y[t]} "
                    f"{amax_x[t] - int(centers[t][0]):+d} "
                    f"{amax_y[t] - int(centers[t][1]):+d} "
                    f"{1 if full[t] else 0}\n")
    with open(os.path.join(cfg.out, "zero_diag.txt"), "w") as f:
        f.write("# x y n_zero n_frames zero_always zero_sometimes\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {int(n_zero_2d[i,j])} {T} "
                        f"{1 if zero_always[i,j] else 0} "
                        f"{1 if zero_sometimes[i,j] else 0}\n")

    # dead-from-data: нуль во ВСЕХ кадрах
    dead_init = read_dead_file(cfg.read_dead, PADS_Y, PADS_X) | zero_always
    if cfg.dead_thresh_eps <= 0.0:
        log("dead: авто-детекция по eps отключена")
    log(f"dead: из данных = {int(zero_always.sum())}, "
        f"итог = {int(dead_init.sum())}")

    log(f"\n=== fit M12 (strict) ===")
    res = fit_m12(data, data_raw, dx, dy, mask, PADS_Y, PADS_X, cfg, log,
                  dead_init=dead_init)
    eps = res["eps"]; A = res["A"]
    dead = res["dead"]; b = res["b"]
    shift_x, shift_y = res["shift"]
    dx_c = res["dx_c"]; dy_c = res["dy_c"]; V_grid0 = res["V_grid"]
    interp = res["interp"] if "interp" in res else None

    # V-выходы (свободная V)
    DXg, DYg = np.meshgrid(dx_c - shift_x, dy_c - shift_y, indexing="xy")
    V_out = V_grid0 / V_grid0.max()
    np.savetxt(os.path.join(cfg.out, "V_dxdy.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), V_out.ravel()],
               header="dx dy V")
    np.savetxt(os.path.join(cfg.out, "V_emp_dxdy.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), V_grid0.ravel()],
               header="dx dy V_emp")

    # V(r)
    t_idx = res["t_idx"]; i_idx = res["i_idx"]; j_idx = res["j_idx"]
    D = res["D"]
    DX = dx[t_idx, i_idx, j_idx] - shift_x
    DY = dy[t_idx, i_idx, j_idx] - shift_y
    R_pt = np.sqrt(DX ** 2 + DY ** 2)
    r_bins = np.linspace(0, cfg.r_max, 31)
    H, _ = np.histogram(R_pt, bins=r_bins, weights=D)
    C, _ = np.histogram(R_pt, bins=r_bins)
    V_emp_r = np.where(C > 0, H / np.maximum(C, 1e-9), 0.0)
    cb = 0.5 * (r_bins[1:] + r_bins[:-1])
    Rg = np.sqrt(DXg ** 2 + DYg ** 2)
    V_r_model = np.zeros_like(cb)
    for k in range(len(cb)):
        m = (Rg >= r_bins[k]) & (Rg < r_bins[k + 1]) & (V_out > 0)
        if m.sum() > 0:
            V_r_model[k] = V_out[m].mean()
    V_r_model /= max(V_r_model.max(), 1e-12)
    good = V_emp_r > 1e-6
    scale = (np.sum(V_emp_r[good] * V_r_model[good]) /
             max(np.sum(V_r_model[good] ** 2), 1e-30)) if good.sum() > 3 else 1.0
    log("\n=== V(r) model vs V_emp(r) ===")
    log(f"{'r':>8s} {'V_emp':>12s} {'V_model*scale':>14s} {'ratio':>10s}")
    for k in range(0, len(cb), 2):
        vm = V_r_model[k] * scale
        ratio = vm / V_emp_r[k] if V_emp_r[k] > 1e-12 else 0.0
        log(f"{cb[k]:8.2f} {V_emp_r[k]:12.4e} {vm:14.4e} {ratio:10.3f}")

    # остатки
    V_tab = res["V_tab"]; Wf = res["Wf"]
    resid = res["resid"]; abs_r = res["resid_abs"]
    rms = float(np.sqrt(np.mean(resid ** 2)))
    sigma_mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    frac_3s = float(np.mean(abs_r > 3 * rms))
    frac_5s = float(np.mean(abs_r > 5 * rms))
    frac_3mad = float(np.mean(abs_r > 3 * sigma_mad))
    frac_5mad = float(np.mean(abs_r > 5 * sigma_mad))
    chi2_total = float(np.sum(Wf * resid ** 2))
    chi2_out5 = float(np.sum(Wf[abs_r > 5 * rms] * resid[abs_r > 5 * rms] ** 2))
    w_ratio = Wf / np.maximum(res["Wf_base"], 1e-30)
    log(f"\n=== диагностика остатков ===")
    log(f"rms = {rms:.4e}, max|r| = {np.max(abs_r):.4e}, sigma_MAD = {sigma_mad:.4e}")
    log(f"frac(|r|>3*rms) = {frac_3s:.3%}, frac(|r|>5*rms) = {frac_5s:.3%}")
    log(f"frac(|r|>3*sigma_MAD) = {frac_3mad:.3%}, "
        f"frac(|r|>5*sigma_MAD) = {frac_5mad:.3%}")
    log(f"chi2_w/M = {chi2_total / D.size:.4e}")
    log(f"chi2: total = {chi2_total:.4e}, outliers = {chi2_out5:.4e} "
        f"({100 * chi2_out5 / max(chi2_total, 1e-30):.1f}%)")
    log(f"Wf/Wf_base: median = {np.median(w_ratio):.3f}, min = {np.min(w_ratio):.3e}")
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
    np.save(os.path.join(cfg.out, "resid_mean.npy"), mean_resid)
    np.save(os.path.join(cfg.out, "resid_abs_mean.npy"), mean_abs)
    np.save(os.path.join(cfg.out, "resid_counts.npy"), cnt_2d)
    with open(os.path.join(cfg.out, "resid_map.txt"), "w") as f:
        f.write("# x y mean_resid mean_abs_resid n_points\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {mean_resid[i,j]:.6e} {mean_abs[i,j]:.6e} "
                        f"{cnt_2d[i,j]}\n")
    flat_order = np.argsort(np.nan_to_num(mean_abs, nan=-1).ravel())[::-1]
    log("\n=== топ-10 пикселей по mean|resid| ===")
    log(f"{'x':>4s} {'y':>4s} {'mean|r|':>12s} {'mean_r':>12s} "
        f"{'n':>6s} {'eps':>10s} {'b':>10s} {'dead':>4s}")
    for k in flat_order[:10]:
        i, j = divmod(int(k), PADS_X)
        log(f"{j:4d} {i:4d} {mean_abs[i,j]:12.4e} {mean_resid[i,j]:12.4e} "
            f"{cnt_2d[i,j]:6d} {eps[i,j]:10.3e} {b[i,j]:10.3e} "
            f"{1 if dead[i,j] else 0:4d}")

    # sigma_stat
    log(f"\n=== sigma_stat (kfold K={cfg.kfold}) ===")
    eps_stat, folds = kfold_stat(res, PADS_Y, PADS_X, cfg, log)

    r_2d = np.sqrt((np.arange(PADS_X)[None, :] - CX) ** 2 +
                   (np.arange(PADS_Y)[:, None] - CY) ** 2)
    eps_stat[r_2d > _R_SIGMA_MAX] = np.nan
    log(f"sigma_stat = nan для r > {_R_SIGMA_MAX}: "
        f"{(r_2d > _R_SIGMA_MAX).sum()} пикселей")

    dead_final = dead | detect_dead(eps, eps_stat, cfg.dead_thresh_eps,
                                    cfg.dead_thresh_stat)
    if int(dead_final.sum()) != int(dead.sum()):
        log(f"dead: уточнено по sigma_stat: {int(dead.sum())} -> "
            f"{int(dead_final.sum())}")
        eps[dead_final] = 0.0; b[dead_final] = 0.0
        eps_stat[dead_final] = np.nan; dead = dead_final

    # sigma_sys/tot не считаем (эталон не используется) -- NaN/sigma_stat
    eps_sys = np.full_like(eps_stat, np.nan)
    eps_tot = eps_stat.copy()

    out = cfg.out
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
    np.savetxt(os.path.join(out, "V_dxdy_amp.txt"),
               np.c_[DXg.ravel(), DYg.ravel(), (V_out * A_mean).ravel()],
               header="dx dy F")
    np.save(os.path.join(out, "bkg.npy"), b)
    np.savetxt(os.path.join(out, "bkg.txt"),
               np.c_[np.indices((PADS_Y, PADS_X))[1].ravel(),
                     np.indices((PADS_Y, PADS_X))[0].ravel(), b.ravel()],
               header="x y b")
    np.save(os.path.join(out, "P_t.npy"), res["P_t"])
    np.save(os.path.join(out, "eps.npy"), eps)
    np.save(os.path.join(out, "A.npy"), A)
    np.save(os.path.join(out, "eps_folds.npy"), folds)
    if cfg.save_dead:
        write_dead_file(cfg.save_dead, dead, PADS_Y, PADS_X)
        log(f"dead: сохранено {int(dead.sum())} пикселей в {cfg.save_dead}")

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

    sx_m, mx, bgx = sigma_1d(P_avg[CY, :], np.arange(PADS_X, dtype=float))
    sy_m, my, bgy = sigma_1d(P_avg[:, CX], np.arange(PADS_Y, dtype=float))
    log(f"\n=== beam sigma ===\nx: sigma={sx_m:.4f}px  mean={mx:.3f}  bg={bgx:.3e}")
    log(f"y: sigma={sy_m:.4f}px  mean={my:.3f}  bg={bgy:.3e}")

    log.save(os.path.join(out, "fit_log.txt"))


if __name__ == "__main__":
    main()
