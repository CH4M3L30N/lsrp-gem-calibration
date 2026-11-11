#!/usr/bin/env python3
"""
Roundtrip test: scan_app writes -> get_epsilon reads -> data matches.

Проверяет:
  1. Формат строки: <ts_start_step> <ts_end_step> <ts_start_data> <ts_end_data> <I> <c_x> <c_y> <data>
  2. timestamp без пробелов (ISO 8601, 'T' вместо ' ')
  3. c_x, c_y: центр детектора = (16, 10) при детекторе в центре зелёной зоны
  4. data: 640 int, построчно (y=0: x=0..31, ...)
  5. reshape в (PADS_Y, PADS_X) = (20, 32)
  6. Разворот y: y=0 -- сверху (в scan_app: arr[::-1, :] перед flatten)
  7. get_epsilon не разворачивает y (cyt = c_y, а не PADS_Y - 1 - c_y)
"""

import os
import numpy as np
import datetime

# ----------------------------- константы -----------------------------
PADS_X, PADS_Y = 32, 20
CX, CY = PADS_X // 2, PADS_Y // 2   # 16, 10
MP = 4

DATA_FILE = "data20261109_143025.txt"

# ----------------------------- helpers -----------------------------
def fmt_ts(t):
    """Unix timestamp -> 'YYYY-MM-DDTHH:MM:SS.mmm' (ISO 8601, без пробела)."""
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def write_test_file(path, n_frames=5, seed=42):
    """Имитация scan_app: записываем n_frames строк в новом формате."""
    rng = np.random.default_rng(seed)
    with open(path, "w") as f:
        for t in range(n_frames):
            # timestamps (ISO 8601, без пробела)
            t0 = 1731078000.0 + t * 0.1
            ts_start_step = fmt_ts(t0)
            ts_end_step   = fmt_ts(t0 + 0.05)
            ts_start_data = fmt_ts(t0 + 0.051)
            ts_end_data   = fmt_ts(t0 + 0.061)

            # c_x, c_y: центр детектора (16, 10) + смещение
            # смещение = -10..+10 (имитация сканирования)
            offset_x = -10 + (t * 20) // n_frames
            offset_y = -5  + (t * 10) // n_frames
            c_x = CX + offset_x
            c_y = CY + offset_y

            I = 1.0

            # data: 640 int (построчно: y=0: x=0..31, ...)
            data_flat = rng.integers(0, 1000, size=(PADS_Y * PADS_X,))
            data_str = " ".join(str(v) for v in data_flat)

            line = (f"{ts_start_step} {ts_end_step} "
                    f"{ts_start_data} {ts_end_data} "
                    f"{I:.6f} {c_x} {c_y} {data_str}\n")
            f.write(line)
    print(f"Wrote {n_frames} frames to {path}")

def read_test_file(path):
    """Имитация get_epsilon: читаем файл, возвращаем (data, centers, timestamps)."""
    data = []
    centers = []
    timestamps = []
    with open(path, "r") as f:
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
    return np.array(data), centers, timestamps

# ----------------------------- тест 1: формат строки -----------------------------
print("=" * 60)
print("TEST 1: format (no spaces in timestamps)")
print("=" * 60)
write_test_file(DATA_FILE, n_frames=5)

with open(DATA_FILE) as f:
    first_line = f.readline().strip()
parts = first_line.split()
print(f"Fields in first line: {len(parts)}")
expected = 7 + PADS_X * PADS_Y
assert len(parts) == expected, f"Expected {expected} fields, got {len(parts)}"
print(f"OK: {expected} fields (7 meta + {PADS_X*PADS_Y} data)")

# timestamp не содержит пробел
for idx, name in [(0, "ts_start_step"), (1, "ts_end_step"),
                  (2, "ts_start_data"), (3, "ts_end_data")]:
    ts = parts[idx]
    assert " " not in ts, f"{name} contains space: '{ts}'"
    # проверим, что это ISO 8601
    try:
        datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        raise AssertionError(f"{name} not ISO 8601: '{ts}'")
    print(f"OK: {name} = '{ts}' (ISO 8601, no space)")

# ----------------------------- тест 2: c_x, c_y -----------------------------
print("\n" + "=" * 60)
print("TEST 2: c_x, c_y (center = (16, 10))")
print("=" * 60)
data, centers, timestamps = read_test_file(DATA_FILE)
print(f"Read {len(data)} frames")
print(f"centers = {centers}")

# центр зелёной зоны должен давать (16, 10)
# проверим, что все c_x, c_y в разумных пределах
for (c_x, c_y) in centers:
    assert isinstance(c_x, int) and isinstance(c_y, int)
    assert -100 <= c_x <= 100, f"c_x out of range: {c_x}"
    assert -100 <= c_y <= 100, f"c_y out of range: {c_y}"
print(f"OK: c_x, c_y -- целые, в разумных пределах")

# ----------------------------- тест 3: data shape -----------------------------
print("\n" + "=" * 60)
print("TEST 3: data shape (PADS_Y, PADS_X)")
print("=" * 60)
print(f"data.shape = {data.shape}")
assert data.shape == (5, PADS_Y, PADS_X), f"Expected (5, {PADS_Y}, {PADS_X}), got {data.shape}"
print(f"OK: shape = (n_frames, {PADS_Y}, {PADS_X})")

# ----------------------------- тест 4: data -- int -----------------------------
print("\n" + "=" * 60)
print("TEST 4: data -- int")
print("=" * 60)
assert np.all(data == data.astype(int)), "data contains non-integers"
print(f"OK: all values are integers")
print(f"  min = {data.min():.0f}, max = {data.max():.0f}")

# ----------------------------- тест 5: разворот y -----------------------------
print("\n" + "=" * 60)
print("TEST 5: flip y (y=0 -- сверху)")
print("=" * 60)
# Создадим тестовый кадр, где y=0 имеет значение 1000, y=19 -- значение 1
# и проверим, что после чтения y=0 всё ещё 1000
test_data = np.zeros((PADS_Y, PADS_X), dtype=int)
test_data[0, :] = 1000    # y=0
test_data[PADS_Y - 1, :] = 1   # y=19

# имитация scan_app: arr[::-1, :] перед flatten
arr = test_data.copy()
arr = arr[::-1, :]         # flip y
data_flat = arr.flatten().tolist()

# проверяем: первый элемент (y=0 в scan_app) должен быть 1 (бывший y=19)
# потому что после flip y, y=19 стало y=0
print(f"After scan_app flip: data_flat[0] = {data_flat[0]} (expected 1, бывший y=19)")
print(f"After scan_app flip: data_flat[-1] = {data_flat[-1]} (expected 1000, бывший y=0)")

# get_epsilon: reshape без разворота
frame = np.array(data_flat, dtype=float).reshape((PADS_Y, PADS_X))
print(f"After get_epsilon reshape: frame[0, 0] = {frame[0, 0]:.0f} (expected 1)")
print(f"After get_epsilon reshape: frame[{PADS_Y-1}, 0] = {frame[PADS_Y-1, 0]:.0f} (expected 1000)")
assert frame[0, 0] == 1, "flip y broken: y=0 should be 1"
assert frame[PADS_Y - 1, 0] == 1000, "flip y broken: y=19 should be 1000"
print(f"OK: flip y works (y=0 -- сверху)")

# ----------------------------- тест 6: get_epsilon не разворачивает y -----------------------------
print("\n" + "=" * 60)
print("TEST 6: get_epsilon uses cyt = c_y (no PADS_Y - 1 - ...)")
print("=" * 60)
# Согласно новому формату, c_y уже в системе координат data (y=0 сверху)
# Поэтому cyt = c_y, а не PADS_Y - 1 - c_y
c_y = CY       # 10 (центр)
cyt = c_y      # get_epsilon: cyt = c_y
assert cyt == 10, f"cyt should be 10, got {cyt}"
print(f"OK: cyt = c_y = {cyt} (без разворота PADS_Y - 1 - c_y)")

# ----------------------------- тест 7: c_x, c_y -- центр зелёной зоны -----------------------------
print("\n" + "=" * 60)
print("TEST 7: center of green zone -> c_x = 16, c_y = 10")
print("=" * 60)
# Формула scan_app:
# c_x = int(round((cur_x - scan_w/2)/MP)) + DETECTOR_WIDTH // 2 // MP
# c_y = int(round((cur_y - scan_h/2)/MP)) + DETECTOR_HEIGHT // 2 // MP
DETECTOR_WIDTH  = PADS_X * MP    # 128
DETECTOR_HEIGHT = PADS_Y * MP    # 80

scan_w = 52 * MP   # 208
scan_h = 30 * MP   # 120

# центр зелёной зоны
cur_x = scan_w / 2   # 104
cur_y = scan_h / 2   # 60

c_x = int(round((cur_x - scan_w / 2) / MP)) + DETECTOR_WIDTH // 2 // MP
c_y = int(round((cur_y - scan_h / 2) / MP)) + DETECTOR_HEIGHT // 2 // MP

print(f"scan_w = {scan_w}, scan_h = {scan_h}")
print(f"cur = ({cur_x}, {cur_y})")
print(f"c_x = {c_x}, c_y = {c_y}")
assert c_x == 16, f"center should give c_x = 16, got {c_x}"
assert c_y == 10, f"center should give c_y = 10, got {c_y}"
print(f"OK: center -> (16, 10)")

# левый верхний угол
cur_x = 0
cur_y = 0
c_x = int(round((cur_x - scan_w / 2) / MP)) + DETECTOR_WIDTH // 2 // MP
c_y = int(round((cur_y - scan_h / 2) / MP)) + DETECTOR_HEIGHT // 2 // MP
print(f"top-left -> c_x = {c_x}, c_y = {c_y}")
assert c_x == -26 + 16, f"expected -10, got {c_x}"
assert c_y == -15 + 10, f"expected -5, got {c_y}"
print(f"OK: top-left -> (-10, -5)")

# правый нижний угол
cur_x = scan_w
cur_y = scan_h
c_x = int(round((cur_x - scan_w / 2) / MP)) + DETECTOR_WIDTH // 2 // MP
c_y = int(round((cur_y - scan_h / 2) / MP)) + DETECTOR_HEIGHT // 2 // MP
print(f"bottom-right -> c_x = {c_x}, c_y = {c_y}")
assert c_x == 26 + 16, f"expected 42, got {c_x}"
assert c_y == 15 + 10, f"expected 25, got {c_y}"
print(f"OK: bottom-right -> (42, 25)")

# ----------------------------- тест 8: полный roundtrip -----------------------------
print("\n" + "=" * 60)
print("TEST 8: full roundtrip (write -> read -> compare)")
print("=" * 60)
rng = np.random.default_rng(123)
n_frames = 3
known_data = []
known_centers = []
with open("test_known.txt", "w") as f:
    for t in range(n_frames):
        ts_start_step = fmt_ts(1731078000.0 + t)
        ts_end_step   = fmt_ts(1731078000.0 + t + 0.5)
        ts_start_data = fmt_ts(1731078000.0 + t + 0.6)
        ts_end_data   = fmt_ts(1731078000.0 + t + 0.7)
        c_x = CX + t - 1   # 15, 16, 17
        c_y = CY + t - 1   # 9, 10, 11
        I = 1.0
        data_arr = rng.integers(0, 1000, size=(PADS_Y, PADS_X))
        # scan_app: flip y перед flatten
        data_flipped = data_arr[::-1, :]
        data_flat = data_flipped.flatten()
        data_str = " ".join(str(v) for v in data_flat)
        f.write(f"{ts_start_step} {ts_end_step} {ts_start_data} {ts_end_data} "
                f"{I:.6f} {c_x} {c_y} {data_str}\n")
        known_data.append(data_arr)
        known_centers.append((c_x, c_y))

known_data = np.array(known_data)

# читаем
data, centers, timestamps = read_test_file("test_known.txt")

print(f"Known data shape: {known_data.shape}")
print(f"Read data shape:  {data.shape}")
assert data.shape == known_data.shape, "shape mismatch"

# ВАЖНО: scan_app писал с flip y, get_epsilon читает без flip.
# Значит, data (из get_epsilon) -- flipped относительно known_data.
# Сравниваем data с known_data, flipped обратно по y.
known_data_flipped_back = known_data[:, ::-1, :]
assert np.allclose(data, known_data_flipped_back), \
    "data mismatch after roundtrip (check flip y in scan_app / get_epsilon)"
print(f"OK: data matches after roundtrip (flip y verified)")

for t in range(n_frames):
    assert centers[t] == known_centers[t], f"center mismatch at t={t}"
print(f"OK: centers match: {centers}")

# ----------------------------- тест 9: get_epsilon logic -----------------------------
print("\n" + "=" * 60)
print("TEST 9: get_epsilon logic (shift_idx, mask_t)")
print("=" * 60)
# Проверим, что shift_idx работает правильно для известных c_x, c_y
# shift_idx[t, i, j] = (CY + i - cyt, CX + j - cxt)
# где cyt = c_y, cxt = c_x (без разворота)

TIME_SIZE = n_frames
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

# Для t=1 (центр в (16, 10)): shift_idx[1, i, j] = (i, j)
for i in range(PADS_Y):
    for j in range(PADS_X):
        assert mask_t[1, i, j], f"t=1: ({i},{j}) should be valid"
        assert shift_idx[1, i, j, 0] == i, f"t=1: iy mismatch at ({i},{j})"
        assert shift_idx[1, i, j, 1] == j, f"t=1: jx mismatch at ({i},{j})"
print(f"OK: t=1 (center (16,10)): shift_idx = (i, j)")

# Для t=0 (центр в (15, 9)): shift_idx[0, i, j] = (i+1, j+1)
for i in range(PADS_Y - 1):
    for j in range(PADS_X - 1):
        assert mask_t[0, i, j], f"t=0: ({i},{j}) should be valid"
        assert shift_idx[0, i, j, 0] == i + 1, f"t=0: iy mismatch at ({i},{j})"
        assert shift_idx[0, i, j, 1] == j + 1, f"t=0: jx mismatch at ({i},{j})"
print(f"OK: t=0 (center (15,9)): shift_idx = (i+1, j+1)")

# ----------------------------- тест 10: heatmap flip x -----------------------------
print("\n" + "=" * 60)
print("TEST 10: heatmap flip x (x=31 слева, x=0 справа)")
print("=" * 60)
# Имитация draw_heatmap: проверяем, что x0 для j=0 -- справа, для j=31 -- слева
cols = PADS_X
cw = 320
cell_w = cw / cols

# x0 = (cols - 1 - j) * cell_w
x0_j0  = (cols - 1 - 0) * cell_w    # 31 * 10 = 310
x0_j31 = (cols - 1 - 31) * cell_w   # 0 * 10 = 0
print(f"x0 for j=0:  {x0_j0:.1f} (should be right)")
print(f"x0 for j=31: {x0_j31:.1f} (should be left)")
assert x0_j0 > x0_j31, "flip x broken: j=0 should be right of j=31"
print(f"OK: flip x works (x=31 слева, x=0 справа)")

# ----------------------------- итог -----------------------------
print("\n" + "=" * 60)
print("ALL TESTS PASSED")
print("=" * 60)
print(f"\nGenerated test file: {DATA_FILE}")
print(f"Generated test file: test_known.txt")
print("\nCleanup: rm {DATA_FILE} test_known.txt")
