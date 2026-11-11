#!/usr/bin/env python3
"""
ScannerCore: чистая логика сканирования (без tkinter).

Содержит:
  - путь сканирования (build_path)
  - движение мотора (move_support)
  - запись данных (save_data)
  - позиции (start, center)
  - состояние (running, paused, position)

GUI (scan_app.py) вызывает методы и обновляет виджеты.
CLI (scan_cli.py) вызывает методы и печатает в консоль.

Не зависит от tkinter. Использует:
  - devices (опционально, для мотора)
  - epics (опционально, для данных)
"""

import os
import math
import time
import datetime
import numpy as np

# ----------------------------- optional hardware -----------------------------
try:
    import devices
    HAVE_DEVICES = True
except ImportError:
    HAVE_DEVICES = False

try:
    import epics
    HAVE_EPICS = True
except ImportError:
    HAVE_EPICS = False

# ----------------------------- constants -----------------------------
PAD_MM = (2.0, 1.0)
PADS_X, PADS_Y = 32, 20
MP = 4
DETECTOR_WIDTH  = PADS_X * MP
DETECTOR_HEIGHT = PADS_Y * MP
SCALE = [1, 2]

# ----------------------------- helpers -----------------------------
def fmt_ts(t):
    """Unix timestamp -> 'YYYY-MM-DDTHH:MM:SS.mmm' (ISO 8601, без пробела)."""
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def epics_ts_to_str(val):
    """EPICS timestamp -> ISO 8601 (или None)."""
    if val is None:
        return None
    if isinstance(val, str):
        return val
    try:
        return fmt_ts(float(val))
    except (ValueError, TypeError):
        return None

# ----------------------------- ScannerCore -----------------------------
class ScannerCore:
    """Чистая логика сканирования (без GUI)."""

    def __init__(self, x_zone, y_zone, x_step, y_step, t_var,
                 cycles=1, both_sides=False, dev=None):
        """
        x_zone, y_zone : размер зоны в моторных шагах (IntVar в GUI)
        x_step, y_step : шаг в моторных шагах
        t_var          : задержка между шагами (сек)
        cycles         : число циклов
        both_sides     : "both sides" (True) или "one side" (False)
        dev            : devices.Support или None
        """
        self.x_zone = x_zone
        self.y_zone = y_zone
        self.x_step = x_step
        self.y_step = y_step
        self.t_var = t_var
        self.cycles = cycles
        self.both_sides = both_sides
        self.dev = dev

        # состояние
        self.running   = False
        self.paused    = False
        self.position  = 0
        self.path      = []

        # текущая позиция (GUI px внутри зоны)
        self.cur_x = 0.0
        self.cur_y = 0.0

        # геометрия (GUI px)
        self.scan_w = x_zone * MP
        self.scan_h = y_zone * MP

        # hardware reference
        self.ref_x = None
        self.ref_y = None
        self.motor_scale = SCALE
        if self.dev is not None:
            self.ref_x = self.dev.status["x"]
            self.ref_y = self.dev.status["y"]
            try:
                self.motor_scale = self.dev.SCALE
            except AttributeError:
                self.motor_scale = SCALE

        # EPICS PVs
        self.pv = None
        self.pv_tsdas = None
        self.pv_tsdae = None
        if HAVE_EPICS:
            try:
                self.pv = epics.PV('lsrp:spot:hist')
            except Exception:
                self.pv = None
            try:
                self.pv_tsdas = epics.PV('lsrp:spot:timestamp_begin')
            except Exception:
                self.pv_tsdas = None
            try:
                self.pv_tsdae = epics.PV('lsrp:spot:timestamp_end')
            except Exception:
                self.pv_tsdae = None

        # data file
        self.data_file = None
        self.data_filepath = None

        # callbacks (для GUI / CLI)
        self.on_step = None        # вызывается после каждого шага
        self.on_status = None      # вызывается при смене статуса
        self.on_data = None        # вызывается при новых данных (arr)
        self.on_finished = None    # вызывается при завершении
        self.on_error = None       # вызывается при ошибке

    # ---------- callbacks ----------
    def _set_status(self, text):
        if self.on_status is not None:
            self.on_status(text)

    def _set_data(self, arr):
        if self.on_data is not None:
            self.on_data(arr)

    def _set_error(self, text):
        if self.on_error is not None:
            self.on_error(text)

    # ---------- path ----------
    def build_path(self):
        xs = max(self.x_step * MP, 1)
        ys = max(self.y_step * MP, 1)
        nx = int(self.scan_w // xs) + 1
        ny = int(self.scan_h // ys) + 1
        n_cycles = max(self.cycles, 1)

        path = []
        for _ in range(n_cycles):
            for j in range(ny):
                cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
                for i in cols:
                    path.append((i * xs, j * ys))
            if self.both_sides:
                for j in range(ny - 1, -1, -1):
                    if (ny - 1) % 2 == 0:
                        cols = range(nx - 1, -1, -1) if j % 2 != 0 else range(nx)
                    else:
                        cols = range(nx) if j % 2 != 0 else range(nx - 1, -1, -1)
                    for i in cols:
                        path.append((i * xs, j * ys))
        return path

    def start_position(self):
        return (0.0, 0.0)

    def center_position(self):
        return (self.scan_w / 2, self.scan_h / 2)

    # ---------- hardware ----------
    def move_support(self, cx, cy):
        if self.dev is None:
            return time.time()
        dx = (cx - self.scan_w / 2) / MP
        dy = (cy - self.scan_h / 2) / MP
        sc = self.motor_scale
        tx = self.ref_x + int(round(dx * PAD_MM[0] / sc[0]))
        ty = self.ref_y - int(round(dy * PAD_MM[1] / sc[1]))
        t_move = time.time()
        try:
            self.dev.move_to(tx, ty)
        except Exception as e:
            self._set_error(f"Hardware error: {e}")
            return None
        return t_move

    # ---------- save_data (returns True if written, False if stale) ----------
    def save_data(self, ts_start_step, ts_end_step):
        if self.data_file is None:
            return False

        if self.pv is None:
            data_flat = [0] * (PADS_X * PADS_Y)
            arr = None
        else:
            value = self.pv.get()
            if value is None:
                data_flat = [0] * (PADS_X * PADS_Y)
                arr = None
            else:
                arr = np.array(value)
                try:
                    arr.resize(PADS_Y, PADS_X)
                    arr = arr[::-1, :]            # flip y
                    data_flat = arr.flatten().astype(int).tolist()
                except Exception:
                    data_flat = [0] * (PADS_X * PADS_Y)
                    arr = None

        ts_start_data = epics_ts_to_str(self.pv_tsdas.get() if self.pv_tsdas else None)
        ts_end_data   = epics_ts_to_str(self.pv_tsdae.get() if self.pv_tsdae else None)

        try:
            ts_end_step_dt = datetime.datetime.fromisoformat(ts_end_step)
            ts_end_data_dt = datetime.datetime.fromisoformat(ts_end_data) if ts_end_data else None
        except (ValueError, TypeError):
            ts_end_data_dt = None

        if ts_end_data_dt is None or ts_end_data_dt <= ts_end_step_dt:
            return False

        if arr is not None:
            self._set_data(arr)

        c_x = int(round((self.cur_x - self.scan_w / 2) / MP)) + DETECTOR_WIDTH // 2 // MP
        c_y = int(round((self.cur_y - self.scan_h / 2) / MP)) + DETECTOR_HEIGHT // 2 // MP

        I = 1.0

        data_str = " ".join(str(v) for v in data_flat)
        line = (f"{ts_start_step} {ts_end_step} "
                f"{ts_start_data} {ts_end_data} "
                f"{I:.6f} {c_x} {c_y} {data_str}\n")
        self.data_file.write(line)
        self.data_file.flush()
        return True

    # ---------- control ----------
    def start(self):
        if self.running or self.paused:
            return

        # новый файл
        if self.data_file is not None:
            self.data_file.close()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_filepath = f"data/data{ts}.txt"
        self.data_file = open(self.data_filepath, "w")

        self.path = self.build_path()
        self.position = 0
        self.running = True
        self.paused = False
        self._set_status("Running")

        # двигаем в стартовую позицию
        self.cur_x, self.cur_y = self.start_position()
        t_move = self.move_support(self.cur_x, self.cur_y)
        if t_move is None:
            self.running = False
            return

        self._step()

    def pause(self):
        if not self.running:
            return
        self.running = False
        self.paused = True
        self._set_status("Paused")

    def stop(self):
        self.running = False
        self.paused = False
        if self.data_file is not None:
            self.data_file.close()
            self.data_file = None
        self._set_status("Stopped")
        # двигаем в центр
        self.move_support(*self.center_position())

    # ---------- scanning ----------
    def _step(self):
        if not self.running:
            return
        if self.position >= len(self.path):
            self.running = False
            self.paused = False
            self._set_status("Finished")
            if self.data_file is not None:
                self.data_file.close()
                self.data_file = None
            if self.on_finished is not None:
                self.on_finished()
            return

        now = time.time()
        ts_start_step = fmt_ts(now - self.t_var)
        ts_end_step   = fmt_ts(now)

        ok = self.save_data(ts_start_step, ts_end_step)
        if not ok:
            self._set_status(f"Waiting for fresh data (pos {self.position})")
            # ждём (через callback / poll)
            if self.on_step is not None:
                self.on_step(self.position, len(self.path), waiting=True)
            # возвращаем управление -- GUI/CLI сам решит, когда вызвать _step снова
            return

        # данные свежие -- двигаем мотор
        self.cur_x, self.cur_y = self.path[self.position]
        t_move = self.move_support(self.cur_x, self.cur_y)
        if t_move is None:
            self.running = False
            self.paused = True
            return

        self.position += 1
        if self.on_step is not None:
            self.on_step(self.position, len(self.path), waiting=False)

    # ---------- poll (для GUI/CLI) ----------
    def poll(self):
        """Вызывается периодически. Если running -- делает шаг."""
        if not self.running:
            return
        self._step()
