#!/usr/bin/env python3
"""
compare_eps.py — сравнение двух epsilon.txt (из get_epsilon.py).

Формат epsilon.txt:
    # x y eps sigma_stat sigma_sys sigma_tot dead_flag b
    0 0 <eps> <sigma_stat> <sigma_sys> <sigma_tot> <dead_flag> <b>

Парсим dead_flag из 7-й колонки (индекс 6). Если колонок меньше 7 —
dead_flag = 0.

При сравнении:
  • исключаем пиксели, которые dead ХОТЯ БЫ В ОДНОМ файле;
  • исключаем пиксели с r > --r-max от центра (по умолчанию 13) —
    это краевые пиксели, где V слабый и eps — артефакт V_emp.

Использование:
    python compare_eps.py \
        --eps-a out_first_strict_s2/epsilon.txt \
        --eps-b out_second_strict_s2/epsilon.txt \
        --r-max 13 \
        --out out_compare_strict_s2_r13
"""

import os
import argparse
import numpy as np


def load_eps(path, PADS_Y, PADS_X, log=print):
    """
    Читает epsilon.txt. Возвращает (eps, dead, has_dead_col).
    dead_flag — 7-я колонка (индекс 6). Если её нет — dead = 0.
    """
    eps = np.full((PADS_Y, PADS_X), np.nan)
    dead = np.zeros((PADS_Y, PADS_X), dtype=bool)
    n_lines = 0
    n_with_dead = 0
    n_cols_max = 0
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
            if not (0 <= y < PADS_Y and 0 <= x < PADS_X):
                continue
            eps[y, x] = v
            n_lines += 1
            n_cols_max = max(n_cols_max, len(p))
            if len(p) >= 7:
                try:
                    if int(p[6]) == 1:
                        dead[y, x] = True
                        n_with_dead += 1
                except ValueError:
                    pass
    has_dead_col = (n_cols_max >= 7)
    log(f"  {path}: {n_lines} строк, dead_flag=1: {n_with_dead}, "
        f"max_cols={n_cols_max}, has_dead_col={has_dead_col}")
    return eps, dead, has_dead_col


def main():
    ap = argparse.ArgumentParser(
        description="Сравнение двух epsilon.txt: corr, rms, карта разности, "
                    "центральная ямка. Исключает dead хотя бы в одном файле "
                    "и краевые пиксели (r > r_max).")
    ap.add_argument("--eps-a", required=True, help="Первый epsilon.txt")
    ap.add_argument("--eps-b", required=True, help="Второй epsilon.txt")
    ap.add_argument("--pads-x", type=int, default=32)
    ap.add_argument("--pads-y", type=int, default=20)
    ap.add_argument("--center-x", type=int, nargs=2, default=(13, 16),
                    help="Диапазон x центральной области")
    ap.add_argument("--center-y", type=int, nargs=2, default=(9, 11),
                    help="Диапазон y центральной области")
    ap.add_argument("--r-max", type=float, default=13.0,
                    help="Исключить пиксели с r > r_max от центра (16, 10). "
                         "Краевые пиксели — артефакт V_emp. "
                         "0 или отрицательное — не исключать.")
    ap.add_argument("--out", default=".", help="Каталог вывода")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    PADS_X, PADS_Y = args.pads_x, args.pads_y
    log_lines = []
    def log(msg):
        print(msg)
        log_lines.append(str(msg))

    log(f"=== compare_eps ===")
    eps_a, dead_a, _ = load_eps(args.eps_a, PADS_Y, PADS_X, log=log)
    eps_b, dead_b, _ = load_eps(args.eps_b, PADS_Y, PADS_X, log=log)
    log(f"A: {args.eps_a}")
    log(f"B: {args.eps_b}")

    # Маска: оба живые, не dead НИ В ОДНОМ
    m = np.isfinite(eps_a) & np.isfinite(eps_b)
    m &= ~dead_a & ~dead_b

    # Исключить краевые пиксели (r > r_max от центра)
    cx_c = PADS_X // 2
    cy_c = PADS_Y // 2
    xx = np.arange(PADS_X, dtype=float)[None, :] - cx_c
    yy = np.arange(PADS_Y, dtype=float)[:, None] - cy_c
    r_2d = np.sqrt(xx ** 2 + yy ** 2)
    if args.r_max > 0:
        m &= r_2d <= args.r_max
        n_r_excluded = int((r_2d > args.r_max).sum())
    else:
        n_r_excluded = 0

    n = int(m.sum())
    n_total = PADS_Y * PADS_X
    log(f"\nмаска: r_max = {args.r_max} (исключено {n_r_excluded} краевых)")
    log(f"общих живых пикселей (не dead, r <= r_max): {n} / {n_total}")
    if n < 10:
        log("мало общих пикселей — сравнение невозможно")
        with open(os.path.join(args.out, "compare_log.txt"), "w") as f:
            f.write("\n".join(log_lines))
        return

    a = eps_a[m]
    b = eps_b[m]
    diff = a - b
    corr = np.corrcoef(a, b)[0, 1]
    rms = np.sqrt(np.mean(diff ** 2))
    log(f"\n=== общая статистика (живые, не dead, r <= r_max) ===")
    log(f"corr = {corr:.5f}")
    log(f"rms(diff) = {rms:.4e}")
    log(f"median(diff) = {np.median(diff):+.4e}")
    log(f"std(diff) = {np.std(diff):.4e}")
    log(f"A: mean={a.mean():.4f}, std={a.std():.4f}, "
        f"min={a.min():.4f}, max={a.max():.4f}")
    log(f"B: mean={b.mean():.4f}, std={b.std():.4f}, "
        f"min={b.min():.4f}, max={b.max():.4f}")

    # Центр
    cx0, cx1 = args.center_x
    cy0, cy1 = args.center_y
    center = np.zeros((PADS_Y, PADS_X), dtype=bool)
    center[cy0:cy1 + 1, cx0:cx1 + 1] = True
    center &= m
    if center.sum() > 0:
        a_c = eps_a[center]
        b_c = eps_b[center]
        log(f"\n=== центр ({cx0}..{cx1}, {cy0}..{cy1}) ===")
        log(f"n_center = {center.sum()}")
        log(f"A: mean={a_c.mean():.4f}, std={a_c.std():.4f}")
        log(f"B: mean={b_c.mean():.4f}, std={b_c.std():.4f}")
        log(f"diff: mean={(a_c - b_c).mean():+.4e}, "
            f"std={(a_c - b_c).std():.4e}")
    else:
        log(f"\n=== центр ({cx0}..{cx1}, {cy0}..{cy1}) ===")
        log("центр пуст (все dead или вне маски)")

    # Периферия (не центр, но внутри r_max)
    periph = m & ~center
    if periph.sum() > 0:
        a_p = eps_a[periph]
        b_p = eps_b[periph]
        log(f"\n=== периферия (не центр, r <= r_max) ===")
        log(f"n_periph = {periph.sum()}")
        log(f"A: mean={a_p.mean():.4f}, std={a_p.std():.4f}")
        log(f"B: mean={b_p.mean():.4f}, std={b_p.std():.4f}")
        log(f"diff: mean={(a_p - b_p).mean():+.4e}, "
            f"std={(a_p - b_p).std():.4e}")

    # Карта разности
    diff_full = eps_a - eps_b
    np.save(os.path.join(args.out, "diff_eps.npy"), diff_full)
    with open(os.path.join(args.out, "diff_eps.txt"), "w") as f:
        f.write("# x y eps_a eps_b diff dead_a dead_b in_mask\n")
        for i in range(PADS_Y):
            for j in range(PADS_X):
                f.write(f"{j} {i} {eps_a[i,j]:.6e} {eps_b[i,j]:.6e} "
                        f"{diff_full[i,j]:.6e} "
                        f"{1 if dead_a[i,j] else 0} "
                        f"{1 if dead_b[i,j] else 0} "
                        f"{1 if m[i,j] else 0}\n")

    # Текстовая карта diff
    log(f"\n=== карта diff (eps_a - eps_b), только в маске, центр [выделен] ===")
    header = "    "
    for j in range(PADS_X):
        header += f"{j:6d}"
    log(header)
    for i in range(PADS_Y):
        row = f"{i:3d} "
        for j in range(PADS_X):
            if dead_a[i, j] or dead_b[i, j]:
                row += "  dead"
            elif not m[i, j]:
                if r_2d[i, j] > args.r_max:
                    row += "   r>"   # вне r_max
                else:
                    row += "   nan"
            else:
                v = diff_full[i, j]
                if abs(v) < 0.05:
                    ch = "  .  "
                else:
                    ch = f"{v:+.2f}"
                if center[i, j]:
                    ch = f"[{ch.strip()}]"[:5]
                row += f"{ch:>6s}"
        log(row)

    # Топ-10 пикселей по |diff| (только в маске)
    diff_abs = np.abs(diff_full)
    diff_abs_masked = np.where(m, diff_abs, -1.0)
    flat_order = np.argsort(diff_abs_masked.ravel())[::-1]
    log(f"\n=== топ-10 пикселей по |eps_a - eps_b| (только в маске) ===")
    log(f"{'x':>4s} {'y':>4s} {'eps_a':>10s} {'eps_b':>10s} {'diff':>10s}")
    k_shown = 0
    for k in flat_order:
        if diff_abs_masked.ravel()[k] <= 0:
            break
        i, j = divmod(int(k), PADS_X)
        log(f"{j:4d} {i:4d} {eps_a[i,j]:10.4e} {eps_b[i,j]:10.4e} "
            f"{diff_full[i,j]:+10.4e}")
        k_shown += 1
        if k_shown >= 10:
            break

    with open(os.path.join(args.out, "compare_log.txt"), "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
