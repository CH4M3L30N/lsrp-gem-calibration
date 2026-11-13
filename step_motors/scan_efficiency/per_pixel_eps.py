#!/usr/bin/env python3
"""
per_pixel_eps.py — per-pixel оценка eps без V_emp.

Для каждого пикселя (i,j):
  eps[i,j] = median_t( data[t,i,j] / (A_t · V(dx,dy)) )
по кадрам, где V(dx,dy) > threshold (пучок рядом с пикселем).

V — параметрическая (гауссова или из get_epsilon.py V_params.txt).
A_t — амплитуда кадра (median по пикселям с V > threshold).

Это даёт eps, не зависящую от V_emp systematic errors.
Если ямка в eps исчезает — она артефакт V_emp.
Если остаётся — реальная неоднородность.
"""

import os
import argparse
import numpy as np


def parse_data(path, PADS_X, PADS_Y):
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
    I0 = float(np.median(I_vals[I_vals > 0]))
    data = data_raw * (I0 / I_vals)[:, None, None]
    centers = np.array(centers, dtype=float)
    return data, data_raw, centers, I0


def parse_v_params(path):
    """Читает V_params.txt из get_epsilon.py.
    Формат: A sx1 sy1 theta_deg eta gamma sx2 sy2 w shift_x shift_y
    или:     A sx sy theta_deg eta gamma shift_x shift_y
    Возвращает dict.
    """
    with open(path) as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    if not lines:
        raise ValueError(f"Empty V_params file: {path}")
    vals = [float(v) for v in lines[0].split()]
    if len(vals) == 12:  # sum2v
        return {
            "A": vals[0], "sx1": vals[1], "sy1": vals[2],
            "theta": np.radians(vals[3]), "eta": vals[4], "gamma": vals[5],
            "sx2": vals[6], "sy2": vals[7], "w": vals[8],
            "shift_x": vals[9], "shift_y": vals[10],
            "sum2v": True,
        }
    elif len(vals) == 8:  # single Voigt
        return {
            "A": vals[0], "sx": vals[1], "sy": vals[2],
            "theta": np.radians(vals[3]), "eta": vals[4], "gamma": vals[5],
            "shift_x": vals[6], "shift_y": vals[7],
            "sum2v": False,
        }
    else:
        raise ValueError(f"Unknown V_params format ({len(vals)} values)")


def _pseudo_voigt(u, v, eta, gamma):
    g = np.exp(-0.5 * (u * u + v * v))
    l = 1.0 / (1.0 + u * u + v * v + gamma * gamma)
    return eta * l + (1.0 - eta) * g


def _gauss(u, v):
    return np.exp(-0.5 * (u * u + v * v))


def V_param(dx, dy, p):
    """Параметрическая V (одиночный Voigt или сумма 2 Voigt)."""
    A = p["A"]
    th = p["theta"]
    ct, st = np.cos(th), np.sin(th)
    if p["sum2v"]:
        u1 = (dx * ct + dy * st) / p["sx1"]
        v1 = (-dx * st + dy * ct) / p["sy1"]
        u2 = (dx * ct + dy * st) / p["sx2"]
        v2 = (-dx * st + dy * ct) / p["sy2"]
        voigt = _pseudo_voigt(u1, v1, p["eta"], p["gamma"])
        gauss = _gauss(u2, v2)
        return A * (p["w"] * voigt + (1.0 - p["w"]) * gauss)
    else:
        u = (dx * ct + dy * st) / p["sx"]
        v = (-dx * st + dy * ct) / p["sy"]
        return A * _pseudo_voigt(u, v, p["eta"], p["gamma"])


def gaussian_V(dx, dy, A, sx, sy):
    return A * np.exp(-0.5 * ((dx / sx) ** 2 + (dy / sy) ** 2))


def per_pixel_eps(data, centers, V_func, V_thresh=0.05,
                  n_min=10, log=print, dead=None):
    """
    data: (T, PADS_Y, PADS_X)
    centers: (T, 2) — c_x, c_y
    V_func: f(dx, dy) -> V (без A_t)
    V_thresh: порог V (доля от максимума), ниже — не используем кадр
    n_min: минимум кадров на пиксель
    dead: (PADS_Y, PADS_X) bool — мёртвые (пропускаем)
    """
    T, PADS_Y, PADS_X = data.shape
    eps_pixel = np.full((PADS_Y, PADS_X), np.nan)
    n_used = np.zeros((PADS_Y, PADS_X), dtype=int)
    A_t = np.zeros(T)

    # Оценка A_t: медиана data / (eps≈1 · V) по пикселям с V > thresh
    for t in range(T):
        cx, cy = centers[t]
        iy, ix = np.mgrid[0:PADS_Y, 0:PADS_X]
        dx = ix - cx
        dy = iy - cy
        V = V_func(dx, dy)
        m = V > V_thresh
        if m.sum() > 10:
            vals = data[t][m] / np.maximum(V[m], 1e-6)
            A_t[t] = np.median(vals)
        else:
            A_t[t] = 1.0
    log(f"A_t: median={np.median(A_t):.3f}, min={A_t.min():.3f}, "
        f"max={A_t.max():.3f}")

    # Per-pixel: медиана data / (A_t · V) по кадрам с V > thresh
    for t in range(T):
        cx, cy = centers[t]
        iy, ix = np.mgrid[0:PADS_Y, 0:PADS_X]
        dx = ix - cx
        dy = iy - cy
        V = V_func(dx, dy)
        m = V > V_thresh
        if m.sum() == 0:
            continue
        vals = data[t][m] / (A_t[t] * np.maximum(V[m], 1e-6))
        # Собираем по пикселям
        for (i, j), v in zip(np.argwhere(m), vals):
            if np.isfinite(v) and v > 0:
                # медиана — накапливаем в списке, но для памяти лучше
                # использовать running median через два массива
                # (здесь для простоты — списки, но T*PADS ~ 693*640 = 443k,
                #  это допустимо)
                pass

    # Простая реализация через списки (медленно, но надёжно)
    from collections import defaultdict
    vals_dict = defaultdict(list)
    for t in range(T):
        cx, cy = centers[t]
        iy, ix = np.mgrid[0:PADS_Y, 0:PADS_X]
        dx = ix - cx
        dy = iy - cy
        V = V_func(dx, dy)
        m = V > V_thresh
        if m.sum() == 0:
            continue
        vals = data[t][m] / (A_t[t] * np.maximum(V[m], 1e-6))
        for (i, j), v in zip(np.argwhere(m), vals):
            if np.isfinite(v) and v > 0:
                vals_dict[(i, j)].append(v)

    for (i, j), lst in vals_dict.items():
        if len(lst) >= n_min:
            if dead is not None and dead[i, j]:
                continue
            eps_pixel[i, j] = np.median(lst)
            n_used[i, j] = len(lst)

    return eps_pixel, n_used, A_t


def compare(eps_a, eps_b, label_a, label_b, log=print):
    """Сравнение двух eps: корреляция, rms(diff), карта разности."""
    m = np.isfinite(eps_a) & np.isfinite(eps_b)
    if m.sum() < 10:
        log(f"compare({label_a}, {label_b}): мало общих пикселей")
        return
    a = eps_a[m]
    b = eps_b[m]
    diff = a - b
    corr = np.corrcoef(a, b)[0, 1]
    rms = np.sqrt(np.mean(diff ** 2))
    log(f"compare({label_a}, {label_b}): n={m.sum()}, "
        f"corr={corr:.5f}, rms(diff)={rms:.4e}, "
        f"median(diff)={np.median(diff):+.4e}, "
        f"std(diff)={np.std(diff):.4e}")


def save_eps(path, eps, PADS_Y, PADS_X, dead=None):
    with open(path, "w") as f:
        f.write("# x y eps dead_flag\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                d = 1 if (dead is not None and dead[i, j]) else 0
                v = eps[i, j]
                if np.isfinite(v):
                    f.write(f"{j} {i} {v:.10e} {d}\n")
                else:
                    f.write(f"{j} {i} nan {d}\n")


def load_eps(path, PADS_Y, PADS_X):
    """Читает epsilon.txt или eps_pixel.txt. Возвращает (eps, dead)."""
    eps = np.full((PADS_Y, PADS_X), np.nan)
    dead = np.zeros((PADS_Y, PADS_X), dtype=bool)
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            p = ln.split()
            if len(p) < 3:
                continue
            try:
                x, y = int(p[0]), int(p[1])
                v = float(p[2])
            except ValueError:
                continue
            if 0 <= y < PADS_Y and 0 <= x < PADS_X:
                eps[y, x] = v
                if len(p) >= 4 and p[3] == "1":
                    dead[y, x] = True
    return eps, dead


def main():
    ap = argparse.ArgumentParser(
        description="Per-pixel оценка eps без V_emp. Сравнение с eps из "
                    "get_epsilon.py и epsilon_real.txt.")
    ap.add_argument("--data", required=True,
                    help="Путь к входному файлу кадров (формат get_epsilon.py)")
    ap.add_argument("--pads-x", type=int, default=32)
    ap.add_argument("--pads-y", type=int, default=20)
    ap.add_argument("--v-params", default=None,
                    help="Путь к V_params.txt из get_epsilon.py. "
                         "Если не задан — используется гауссова V с "
                         "параметрами --v-sigma.")
    ap.add_argument("--v-sigma", type=float, nargs=2, default=(3.0, 2.0),
                    help="σx, σy для гауссовой V (если --v-params не задан)")
    ap.add_argument("--v-amp", type=float, default=1.0,
                    help="Амплитуда гауссовой V (если --v-params не задан)")
    ap.add_argument("--v-thresh", type=float, default=0.05,
                    help="Порог V (доля от максимума)")
    ap.add_argument("--n-min", type=int, default=10,
                    help="Минимум кадров на пиксель")
    ap.add_argument("--read-dead", default=None,
                    help="Файл маски мёртвых (x y flag) — опционально")
    ap.add_argument("--eps-ref", default=None,
                    help="eps из get_epsilon.py (epsilon.txt) для сравнения")
    ap.add_argument("--eps-real", default=None,
                    help="epsilon_real.txt для сравнения")
    ap.add_argument("--out", default=".", help="Каталог вывода")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    PADS_X, PADS_Y = args.pads_x, args.pads_y
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log(f"Read {args.data}")
    data, data_raw, centers, I0 = parse_data(args.data, PADS_X, PADS_Y)
    T = data.shape[0]
    log(f"  frames={T}, I0={I0:.4f}")
    log(f"  c_x: mean={centers[:,0].mean():.2f}, "
        f"min={centers[:,0].min():.0f}, max={centers[:,0].max():.0f}")
    log(f"  c_y: mean={centers[:,1].mean():.2f}, "
        f"min={centers[:,1].min():.0f}, max={centers[:,1].max():.0f}")

    # V
    if args.v_params:
        log(f"V: параметрическая из {args.v_params}")
        vp = parse_v_params(args.v_params)
        shift_x = vp.get("shift_x", 0.0)
        shift_y = vp.get("shift_y", 0.0)
        def V_func(dx, dy):
            return V_param(dx - shift_x, dy - shift_y, vp)
        log(f"  A={vp['A']:.4f}, shift=({shift_x:.3f}, {shift_y:.3f})")
    else:
        log(f"V: гауссова σ=({args.v_sigma[0]}, {args.v_sigma[1]}), "
            f"A={args.v_amp}")
        def V_func(dx, dy):
            return gaussian_V(dx, dy, args.v_amp,
                              args.v_sigma[0], args.v_sigma[1])

    # Мёртвые
    dead = np.zeros((PADS_Y, PADS_X), dtype=bool)
    if args.read_dead and os.path.exists(args.read_dead):
        with open(args.read_dead) as f:
            for ln in f:
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
        log(f"dead: {int(dead.sum())} пикселей из {args.read_dead}")

    # Per-pixel eps
    log("\n=== per-pixel eps ===")
    eps_pixel, n_used, A_t = per_pixel_eps(
        data, centers, V_func, V_thresh=args.v_thresh,
        n_min=args.n_min, log=log, dead=dead)

    n_ok = int(np.isfinite(eps_pixel).sum())
    log(f"восстановлено пикселей: {n_ok} / {PADS_Y * PADS_X}")
    log(f"eps_pixel: min={np.nanmin(eps_pixel):.4f}, "
        f"max={np.nanmax(eps_pixel):.4f}, "
        f"mean={np.nanmean(eps_pixel):.4f}, "
        f"std={np.nanstd(eps_pixel):.4f}")

    # Сохраняем
    save_eps(os.path.join(args.out, "eps_pixel.txt"),
             eps_pixel, PADS_Y, PADS_X, dead=dead)
    np.save(os.path.join(args.out, "eps_pixel.npy"), eps_pixel)
    np.save(os.path.join(args.out, "n_used.npy"), n_used)
    np.save(os.path.join(args.out, "A_t.npy"), A_t)

    # Сравнение с eps из get_epsilon.py
    if args.eps_ref and os.path.exists(args.eps_ref):
        log(f"\n=== сравнение с {args.eps_ref} ===")
        eps_ref, dead_ref = load_eps(args.eps_ref, PADS_Y, PADS_X)
        compare(eps_pixel, eps_ref, "per-pixel", "get_epsilon", log=log)
        # Карта разности
        diff = eps_pixel - eps_ref
        np.save(os.path.join(args.out, "diff_perpixel_vs_geteps.npy"), diff)
        with open(os.path.join(args.out, "diff_map.txt"), "w") as f:
            f.write("# x y eps_pixel eps_geteps diff\n")
            for i in range(PADS_Y):
                for j in range(PADS_X):
                    a = eps_pixel[i, j]
                    b = eps_ref[i, j]
                    f.write(f"{j} {i} {a:.6e} {b:.6e} {a-b:.6e}\n")

    # Сравнение с epsilon_real.txt
    if args.eps_real and os.path.exists(args.eps_real):
        log(f"\n=== сравнение с {args.eps_real} ===")
        eps_real, _ = load_eps(args.eps_real, PADS_Y, PADS_X)
        compare(eps_pixel, eps_real, "per-pixel", "real", log=log)
        compare(eps_real, eps_pixel, "real", "per-pixel", log=log)

    with open(os.path.join(args.out, "per_pixel_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
