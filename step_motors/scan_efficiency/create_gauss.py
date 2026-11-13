import numpy as np
import random
import os

place_x = 32
place_y = 20


def gaussian(x, y, mean_x, mean_y, sigma_x, sigma_y, count_num):
    d_sqx = (x - mean_x) ** 2
    d_sqy = (y - mean_y) ** 2
    gauss = count_num * np.exp(
        -(d_sqx / (2.0 * sigma_x ** 2) + d_sqy / (2.0 * sigma_y ** 2))
    ) / ((2 * np.pi) ** 0.5 * sigma_x * sigma_y)
    return gauss


# ---------- eps ----------
epsilon = np.zeros((place_y, place_x))
for i in range(place_y):
    for j in range(place_x):
        epsilon[i, j] = random.uniform(0.5, 1.0)

# нормировка на среднее
epsilon_mean = epsilon.mean()
epsilon /= epsilon_mean

# сохраняем эталон в формате x y value (как ждёт get_epsilon.py)
with open("epsilon_real.txt", "w+") as f:
    for i in range(place_y):
        for j in range(place_x):
            f.write(f"{j} {i} {epsilon[i, j]:.10e}\n")

# ---------- генерация кадров ----------
# Реалистичные параметры пучка:
sigma_x = 3.0
sigma_y = 2.0
count_num = 10000.0   # амплитуда (в центре mu ~ 10000/(2π·3·2) ~ 265)
I_val = 1.5           # ток (фиктивный, но реалистичный)

cx, cy = 0, 0
stepx = 1

# Сколько кадров: (place_x+1) * (place_y+1) = 33 * 21 = 693 (как в реальных данных)
n_frames = (place_x + 1) * (place_y + 1)

out_path = "data/sim_data.txt"
os.makedirs("data", exist_ok=True)

centers = []
lines = []
t0 = 0.0
dt = 0.01

for n in range(n_frames):
    # генерируем кадр с текущим центром (cx, cy)
    data = np.zeros((place_y, place_x))
    for i in range(place_y):
        for j in range(place_x):
            mu = gaussian(j, i, cx, cy, sigma_x, sigma_y, count_num)
            data[i, j] = np.random.poisson(max(mu, 0.0)) * epsilon[i, j]

    # сохраняем центры
    centers.append((cx, cy))

    # формируем строку в формате get_epsilon.py:
    # timestamp_start timestamp_end timestamp_mid timestamp_end2 I c_x c_y v0 ... v639
    ts0 = f"2026-01-01T00:00:{t0:06.3f}"
    ts1 = f"2026-01-01T00:00:{t0 + dt:06.3f}"
    ts2 = f"2026-01-01T00:00:{t0 + dt:06.3f}"
    ts3 = f"2026-01-01T00:00:{t0 + 2 * dt:06.3f}"
    vals = " ".join(f"{data[i, j]:.6f}" for i in range(place_y) for j in range(place_x))
    line = f"{ts0} {ts1} {ts2} {ts3} {I_val:.6f} {cx} {cy} {vals}"
    lines.append(line)

    t0 += dt

    # следующий центр
    cx += stepx
    if cx > place_x:
        cx = 0
        cy += 1

# записываем все кадры в один файл
with open(out_path, "w+") as f:
    for line in lines:
        f.write(line + "\n")

# записываем centers отдельно (для справки)
with open("data/sim_centers.txt", "w+") as f:
    for c in centers:
        f.write(f"{c[0]} {c[1]}\n")

print(f"Сгенерировано {len(lines)} кадров → {out_path}")
print(f"Эталон eps → epsilon_real.txt")
print(f"Центры → data/sim_centers.txt")
print(f"Параметры: sigma_x={sigma_x}, sigma_y={sigma_y}, "
      f"count_num={count_num}, I={I_val}")
print(f"Диапазон eps: [{epsilon.min():.4f}, {epsilon.max():.4f}], "
      f"mean={epsilon.mean():.4f}, std={epsilon.std():.4f}")
