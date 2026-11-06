#!/usr/bin/env python3
"""
Combined scanner: GUI animation + stepper-motor control + live heatmap.

Buttons:
  start -- moves detector to top-left corner (Wait), then scans
  pause -- pauses scanning
  stop  -- returns detector to the center of the green zone

Position (in GUI) is measured from the top-left corner of the scan zone (0, 0)
in MOTOR STEPS (1 step = MP px in GUI).

Bottom canvas: heatmap of a 20x32 array (zeros for now).
"""

from tkinter import *
from tkinter import ttk
import math
import time
import os, datetime
import numpy as np

# ----------------------------- optional hardware -----------------------------
try:
    import devices
    HAVE_DEVICES = True
except ImportError:
    HAVE_DEVICES = False
    print("WARNING: 'devices' not found -- DEMO mode (no motors)")

try:
    import epics
    HAVE_EPICS = True
except ImportError:
    HAVE_EPICS = False
    print("WARNING: 'epics' not found -- save_data will be a no-op")

# devices.py reads settings.cfg by relative path
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------- constants -----------------------------
CAN_WIDTH       = 370
CAN_HEIGHT      = 300
MP              = 4

DETECTOR_WIDTH  = 32 * MP
DETECTOR_HEIGHT = 20 * MP
GREEN_WIDTH     = 20 * MP
GREEN_HEIGHT    = 10 * MP

HEAT_COLS       = 32
HEAT_ROWS       = 20

# all sizes in the program are in pads; pad size in mm (x, y)
PAD_MM = (2.0, 1.0)
PADS_X, PADS_Y = 32, 20

SCALE = [1, 2]   # mm per motor step (fallback if devices not available)

# ----------------------------- Scanner -----------------------------
class Scanner:
    def __init__(self, root, canvas, detector_item, green_item, scan_item,
                 x_var, y_var, xstep_var, ystep_var, t_var, cycles_var,
                 status_var, modes_var, posx_label, posy_label, time_label, dev=None):
        self.root = root
        self.canvas = canvas
        self.detector_item = detector_item
        self.green_item = green_item
        self.scan_item = scan_item
        self.x_var = x_var
        self.y_var = y_var
        self.xstep_var = xstep_var
        self.ystep_var = ystep_var
        self.t_var = t_var
        self.cycles_var = cycles_var
        self.status_var = status_var
        self.modes_var = modes_var
        self.posx_label = posx_label
        self.posy_label = posy_label
        self.time_label = time_label
        self.dev = dev

        self.running   = False
        self.paused    = False
        self.returning = False
        self.waiting   = False
        self.position  = 0
        self.path      = []
        self.after_id  = None

        # timing
        self.step_times = []
        self.last_step_time = None
        self.total_time = 0.0
        self.remaining_s = 0.0

        # current detector center (GUI px within blue zone)
        self.cur_x = 0.0
        self.cur_y = 0.0

        # geometry
        self.scan_x0 = 0.0
        self.scan_y0 = 0.0
        self.scan_w  = 0.0
        self.scan_h  = 0.0

        # ----------------------------- hardware reference -----------------------------
        # Equipment: the support position at program start = detector center on the beam
        # (center of the scan zone); set in advance via detector_support.
        # This mirrors nik_scan_app.py.
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
            print(f"Device connected: x = {self.ref_x}, y = {self.ref_y}, "
                  f"SCALE = {self.motor_scale}")

        # EPICS PV for save_data (same as nik_scan_app.py)
        self.pv = None
        if HAVE_EPICS:
            try:
                self.pv = epics.PV('lsrp:spot:hist')
            except Exception as e:
                print(f"WARNING: epics.PV init failed ({e}) -- save_data disabled")
                self.pv = None

        # centers file (opened lazily in start())
        self.centers_file = None

    # ---------- geometry ----------
    def update_geometry(self):
        w = max(self.x_var.get() * MP, 10)
        h = max(self.y_var.get() * MP, 10)
        self.scan_w = w
        self.scan_h = h
        self.scan_x0 = (CAN_WIDTH - w) / 2
        self.scan_y0 = (CAN_HEIGHT - h) / 2

        self.canvas.coords(self.scan_item,
                           self.scan_x0, self.scan_y0,
                           self.scan_x0 + w, self.scan_y0 + h)

        gx = self.scan_x0 + (w - GREEN_WIDTH) / 2
        gy = self.scan_y0 + (h - GREEN_HEIGHT) / 2
        self.canvas.coords(self.green_item,
                           gx, gy, gx + GREEN_WIDTH, gy + GREEN_HEIGHT)

        self.draw_detector(self.cur_x, self.cur_y)

    def draw_detector(self, cx, cy):
        canvas_cx = self.scan_x0 + cx
        canvas_cy = self.scan_y0 + cy
        hw = DETECTOR_WIDTH / 2
        hh = DETECTOR_HEIGHT / 2
        self.canvas.coords(self.detector_item,
                           canvas_cx - hw, canvas_cy - hh,
                           canvas_cx + hw, canvas_cy + hh)
        # position in MOTOR STEPS measured from top-left corner of scan zone
        x_steps = cx / MP
        y_steps = cy / MP
        self.posx_label.config(text=f"x={x_steps:.0f}")
        self.posy_label.config(text=f"y={y_steps:.0f}")

    # ---------- hardware (per nik_scan_app.py) ----------
    def move_support(self, cx, cy):
        """Move the support so that the detector center is at (cx, cy) of the zone (canvas px).
        Returns the movement start time, or None on error."""
        if self.dev is None:
            return time.time()   # demo: pretend movement happened

        dx = (cx - self.scan_w / 2) / MP       # detector center offset from beam, pads
        dy = (cy - self.scan_h / 2) / MP       # canvas: y down, support: y up
        sc = self.motor_scale
        tx = self.ref_x + int(round(dx * PAD_MM[0] / sc[0]))
        ty = self.ref_y - int(round(dy * PAD_MM[1] / sc[1]))
        t_move = time.time()
        try:
            self.dev.move_to(tx, ty)
        except Exception as e:
            self.status_var.set(f"Hardware error: {e}")
            return None
        return t_move

    # ---------- save_data (per nik_scan_app.py) ----------
    def save_data(self, time_idx):
        """Read lsrp:spot:hist, write data/data{N}.txt (32x20) and data/centers.txt."""
        if self.pv is None:
            return   # no epics -- nothing to save

        value = self.pv.get()
        if value is None:
            return

        np_value = np.array(value)
        # expected 20*32 = 640 values; resize to (20, 32)
        try:
            np_value.resize(20, 32)
        except Exception as e:
            print(f"save_data: resize failed ({e})")
            return

        os.makedirs("data", exist_ok=True)
        with open(f"data/data{time_idx}.txt", "w+") as f:
            for i in range(20):
                for j in range(32):
                    f.write(f"{j} {i} {np_value[i][j]}\n")
                    
        draw_heatmap(heat_canvas, np_value)

        if self.centers_file is not None:
            self.centers_file.write(f"{self.cur_x//MP} {self.cur_y//MP}\n")
            self.centers_file.flush()

    # ---------- path ----------
    def build_path(self):
        xs = max(self.xstep_var.get() * MP, 1)
        ys = max(self.ystep_var.get() * MP, 1)
        nx = int(self.scan_w // xs) + 1
        ny = int(self.scan_h // ys) + 1
        n_cycles = max(self.cycles_var.get(), 1)
        both_sides = (self.modes_var.get() == "both sides")

        path = []
        for _ in range(n_cycles):
            for j in range(ny):
                cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
                for i in cols:
                    path.append((i * xs, j * ys))
            if both_sides:
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

    # ---------- control ----------
    def start(self):
        if self.running or self.returning or self.waiting:
            return
        if self.paused:
            self.paused = False
            self.running = True
            self.status_var.set("Running")
            self.step()
            return

        # open centers file (truncate) for a new scan
        os.makedirs("data", exist_ok=True)
        if self.centers_file is not None:
            self.centers_file.close()
        self.centers_file = open("data/centers.txt", "w")

        self.path = self.build_path()
        self.position = 0
        self.step_times = []
        self.last_step_time = None
        self.total_time = 0.0
        self.remaining_s = 0.0

        self.waiting = True
        self.status_var.set("Wait")
        self._move_to_start()

    def _move_to_start(self):
        if not self.waiting:
            return
        tx, ty = self.start_position()
        dx = tx - self.cur_x
        dy = ty - self.cur_y
        dist = math.hypot(dx, dy)

        step_len = MP
        if dist <= step_len:
            self.cur_x, self.cur_y = tx, ty
            self.draw_detector(self.cur_x, self.cur_y)
            t_move = self.move_support(tx, ty)
            if t_move is None:
                self.waiting = False
                return
            self.waiting = False
            self.running = True
            self.status_var.set("Running")
            self.last_step_time = time.time()
            self.step()
            return

        self.cur_x += dx / dist * step_len
        self.cur_y += dy / dist * step_len
        self.draw_detector(self.cur_x, self.cur_y)
        self.status_var.set("Wait")

        delay_ms = max(1, int(self.t_var.get() * 1000))
        self.after_id = self.root.after(delay_ms, self._move_to_start)

    def pause(self):
        if not self.running:
            return
        self.running = False
        self.paused = True
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.status_var.set("Paused")

    def stop(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.paused = False
        self.waiting = False
        self.returning = True
        self.status_var.set("Returning to center...")
        # hardware: move support back to center immediately (like nik_scan_app.stop)
        self.move_support(*self.center_position())
        self._return_step()

    # ---------- scanning ----------
    def step(self):
        if not self.running:
            return
        if self.position >= len(self.path):
            self.running = False
            self.paused = False
            self.status_var.set("Finished")
            if len(self.step_times) > 0:
                avg_step = sum(self.step_times) / len(self.step_times)
            else:
                avg_step = 0.0
            self.total_time = avg_step * len(self.path)
            self.time_label.config(text=self._format_time(self.total_time))
            if self.centers_file is not None:
                self.centers_file.close()
                self.centers_file = None
            return

        now = time.time()
        if self.last_step_time is not None and len(self.step_times) < 10:
            self.step_times.append(now - self.last_step_time)
        self.last_step_time = now

        # save current measurement before moving (same as nik_scan_app.py)
        self.save_data(self.position)

        self.cur_x, self.cur_y = self.path[self.position]
        self.draw_detector(self.cur_x, self.cur_y)
        self.root.update_idletasks()

        t_move = self.move_support(self.cur_x, self.cur_y)
        if t_move is None:
            # hardware error: pause; start will retry this point
            self.running = False
            self.paused = True
            return

        self.position += 1
        self.status_var.set(f"Running {self.position}/{len(self.path)}")

        if len(self.step_times) >= 10:
            avg_step = sum(self.step_times) / len(self.step_times)
            remaining_steps = len(self.path) - self.position
            self.remaining_s = avg_step * remaining_steps
            self.time_label.config(text=self._format_time(self.remaining_s))
        elif len(self.step_times) > 0:
            self.time_label.config(text="00:00.000 (measuring...)")

        delay_ms = max(1, int(self.t_var.get() * 1000))
        self.after_id = self.root.after(delay_ms, self.step)

    @staticmethod
    def _format_time(seconds):
        total = int(seconds * 1000)
        mm = total // 60000
        ss = (total % 60000) // 1000
        ms = total % 1000
        return f"{mm:02d}:{ss:02d}.{ms:03d}"

    # ---------- return to center ----------
    def _return_step(self):
        if not self.returning:
            return
        tx, ty = self.center_position()
        dx = tx - self.cur_x
        dy = ty - self.cur_y
        dist = math.hypot(dx, dy)

        step_len = MP
        if dist <= step_len:
            self.cur_x, self.cur_y = tx, ty
            self.draw_detector(self.cur_x, self.cur_y)
            self.returning = False
            self.status_var.set("Stopped (at center)")
            self.time_label.config(text="00:00.000")
            return

        self.cur_x += dx / dist * step_len
        self.cur_y += dy / dist * step_len
        self.draw_detector(self.cur_x, self.cur_y)
        self.status_var.set(f"Returning... ({dist:.1f} px)")

        # support is already back at center (move_support called in stop());
        # here only animation
        delay_ms = 10
        self.after_id = self.root.after(delay_ms, self._return_step)


# ----------------------------- heatmap -----------------------------
def draw_heatmap(canvas, data_2d):
    """Draw 2D array as a heatmap filling the whole canvas (no white borders)."""
    canvas.delete("heat")
    rows = len(data_2d)
    cols = len(data_2d[0]) if rows > 0 else 0
    if rows == 0 or cols == 0:
        return

    cw = int(canvas["width"])
    ch = int(canvas["height"])
    cell_w = cw / cols
    cell_h = ch / rows

    vmax = max(max(row) for row in data_2d)
    if vmax <= 0:
        vmax = 1.0

    for i in range(rows):
        for j in range(cols):
            v = data_2d[i][j] / vmax
            r = int(255 * v)
            b = int(255 * (1 - v))
            color = f"#{r:02x}00{b:02x}"
            x0 = j * cell_w
            y0 = i * cell_h
            canvas.create_rectangle(x0, y0, x0 + cell_w, y0 + cell_h,
                                    fill=color, outline="", tags="heat")


# ----------------------------- main window -----------------------------
root = Tk()
root.title("Efficiency test")
root.geometry("920x760")
root.resizable(False, False)

x_var      = IntVar(value=52)
y_var      = IntVar(value=30)
xstep_var  = IntVar(value=1)
ystep_var  = IntVar(value=1)
t_var      = DoubleVar(value=0.01)
cycles_var = IntVar(value=1)
modes      = ["one side", "both sides"]
modes_var  = StringVar(value=modes[0])
status_var = StringVar(value="Not ready")

# ============================================================
# LEFT side: two canvases (top = scan, bottom = heatmap)
# ============================================================
pos_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
pos_canvas.place(x=10, y=10)

scan_item     = pos_canvas.create_rectangle(0, 0, 1, 1, fill="blue")
green_item    = pos_canvas.create_rectangle(0, 0, 1, 1, fill="green")
detector_item = pos_canvas.create_rectangle(0, 0, 1, 1, fill="grey")

heat_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
heat_canvas.place(x=10, y=320)

heat_data = [[0.0] * HEAT_COLS for _ in range(HEAT_ROWS)]
draw_heatmap(heat_canvas, heat_data)

# ============================================================
# RIGHT side: controls
# ============================================================
COL_X_LABEL = 430
COL_X_ENTRY = 460
COL_X_POS   = 620
COL_W_ENTRY = 60

# ---- size ----
ttk.Label(root, text="size (motor steps)").place(x=COL_X_LABEL, y=10)

ttk.Label(root, text="x").place(x=COL_X_LABEL, y=50)
ttk.Entry(textvariable=x_var, width=8).place(x=COL_X_ENTRY, y=50)
posx_label = ttk.Label(root, text="x=0")
posx_label.place(x=COL_X_POS, y=50)

ttk.Label(root, text="y").place(x=COL_X_LABEL, y=90)
ttk.Entry(textvariable=y_var, width=8).place(x=COL_X_ENTRY, y=90)
posy_label = ttk.Label(root, text="y=0")
posy_label.place(x=COL_X_POS, y=90)

# ---- step ----
ttk.Label(root, text="step (motor steps)").place(x=COL_X_LABEL, y=140)

ttk.Label(root, text="x").place(x=COL_X_LABEL, y=180)
ttk.Entry(textvariable=xstep_var, width=8).place(x=COL_X_ENTRY, y=180)

ttk.Label(root, text="y").place(x=COL_X_LABEL, y=220)
ttk.Entry(textvariable=ystep_var, width=8).place(x=COL_X_ENTRY, y=220)

# ---- t ----
ttk.Label(root, text="t (s)").place(x=COL_X_LABEL, y=260)
ttk.Entry(textvariable=t_var, width=8).place(x=COL_X_ENTRY, y=260)

# ---- time ----
ttk.Label(root, text="time:").place(x=COL_X_LABEL, y=300)
time_label = ttk.Label(root, text="00:00.000")
time_label.place(x=COL_X_ENTRY + 20, y=300)

# ============================================================
# BUTTONS
# ============================================================
BTN_X = COL_X_LABEL
ttk.Button(root, text="start").place(x=BTN_X, y=360)
ttk.Button(root, text="pause").place(x=BTN_X, y=400)
ttk.Button(root, text="stop").place(x=BTN_X, y=440)

# ============================================================
# MODE / CYCLES / STATUS
# ============================================================
BLOCK_Y = 520

ttk.Label(root, text="mode").place(x=COL_X_LABEL, y=BLOCK_Y)
ttk.Combobox(textvariable=modes_var, values=modes,
             state="readonly", width=14).place(x=COL_X_ENTRY + 30, y=BLOCK_Y)

ttk.Label(root, text="cycles").place(x=COL_X_LABEL, y=BLOCK_Y + 40)
ttk.Entry(textvariable=cycles_var, width=8).place(x=COL_X_ENTRY + 30, y=BLOCK_Y + 40)

ttk.Label(root, text="status").place(x=COL_X_LABEL, y=BLOCK_Y + 80)
ttk.Entry(textvariable=status_var, state="readonly", width=24).place(x=COL_X_ENTRY + 30, y=BLOCK_Y + 80)

# ============================================================
# hardware (optional)
# ============================================================
dev = None
if HAVE_DEVICES:
    try:
        dev = devices.Support("Support")
        print(f"Device connected: x = {dev.status['x']}, y = {dev.status['y']}")
    except Exception as e:
        print(f"WARNING: device init failed ({e}) -- DEMO mode")
        dev = None
else:
    print("DEMO mode: no motors")

if dev is None:
    ttk.Label(root, text="(demo: no motors)", foreground="red").place(x=COL_X_LABEL, y=BLOCK_Y + 120)

# ============================================================
# scanner
# ============================================================
scanner = Scanner(root, pos_canvas, detector_item, green_item, scan_item,
                  x_var, y_var, xstep_var, ystep_var, t_var, cycles_var,
                  status_var, modes_var, posx_label, posy_label, time_label, dev=dev)

# attach commands to buttons
for child in root.winfo_children():
    if isinstance(child, ttk.Button):
        txt = child.cget("text")
        if txt == "start":
            child.config(command=scanner.start)
        elif txt == "pause":
            child.config(command=scanner.pause)
        elif txt == "stop":
            child.config(command=scanner.stop)

scanner.update_geometry()

scanner.cur_x, scanner.cur_y = scanner.center_position()
scanner.draw_detector(scanner.cur_x, scanner.cur_y)

x_var.trace_add("write", lambda *_: scanner.update_geometry())
y_var.trace_add("write", lambda *_: scanner.update_geometry())

# ============================================================
# run
# ============================================================
root.mainloop()
