#!/usr/bin/env python3
"""
Program scans reversed compton laser beam, has two windows:
  current scanning preview (upper canvas)
  current data preview (lower canvas)
  
also shows remaining time after 10 steps

Buttons:
  start -- moves detector to top-left corner (Wait), then scans, also resumes scan after pause
  pause -- pauses scanning
  stop  -- returns detector to the center of the green zone
  
saves data only if step finished before data collection

Data format: data<YYYYMMDD_HHMMSS>.txt
  <ts_start_step> <ts_end_step> <ts_start_data> <ts_end_data> <I> <c_x> <c_y> <data>
  ts_* : "YYYY-MM-DDTHH:MM:SS.mmm"
  data: 640 int
  c_x, c_y: laser center in the system of the detector
"""

from tkinter import *
from tkinter import ttk
import math
import time
import os, datetime
import numpy as np

#                   check if devices or epics are working
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

os.chdir(os.path.dirname(os.path.abspath(__file__)))

#                             global constants

# canvas real size
CAN_WIDTH       = 370
CAN_HEIGHT      = 300

MP              = 4 # multiplier for all things in the app (to count in pads and not pixels)

# detector size
DETECTOR_WIDTH  = 32 * MP
DETECTOR_HEIGHT = 20 * MP

# laser size
GREEN_WIDTH     = 20 * MP
GREEN_HEIGHT    = 10 * MP

# all sizes in the program are in pads; pad size in mm (x, y)
PADS_X, PADS_Y = 32, 20
SCALE = [2., 1.]   # mm per motor step (fallback if devices not available)

#                        helping functions
def fmt_ts(t):
    #  YYYY-MM-DDTHH:MM:SS.mmm
    return datetime.datetime.fromtimestamp(t).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]

def epics_ts_to_str(val):
    #  YYYY-MM-DDTHH:MM:SS.mmm
    if val is None:
        return None
    if isinstance(val, str):
        return val
    try:
        return fmt_ts(float(val))
    except (ValueError, TypeError):
        return None

#                      Scanner class (GUI + logic)
class Scanner:
    def __init__(self, root, canvas, detector_item, green_item, scan_item,
                 x_var, y_var, xstep_var, ystep_var, t_var, cycles_var,
                 status_var, modes_var, posx_label, posy_label, time_label, dev=None):
        
        #               variables for scanning
        # basic app parameters
        self.root = root
        self.canvas = canvas
        
        # painted zones (upper canvas)
        self.detector_item = detector_item
        self.green_item = green_item
        self.scan_item = scan_item
        
        # changable parameters
        self.x_var = x_var
        self.y_var = y_var
        self.xstep_var = xstep_var
        self.ystep_var = ystep_var
        self.t_var = t_var
        self.cycles_var = cycles_var
        self.status_var = status_var
        self.modes_var = modes_var
        
        # scanning parameters
        self.posx_label = posx_label
        self.posy_label = posy_label
        self.time_label = time_label
        self.dev = dev
        self.position  = 0
        self.path      = []
        self.after_id  = None

        # statuses of the scanning
        self.running   = False
        self.paused    = False
        self.returning = False
        self.waiting   = False

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

        # configuring devices
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

        # EPICS PVs
        self.pv = None
        self.pv_tsdas = None
        self.pv_tsdae = None
        if HAVE_EPICS:
            try:
                self.pv = epics.PV('lsrp:spot:hist')
            except Exception as e:
                print(f"WARNING: epics.PV('lsrp:spot:hist') failed ({e})")
                self.pv = None
            try:
                self.pv_tsdas = epics.PV('lsrp:spot:timestamp_begin')
            except Exception as e:
                print(f"WARNING: epics.PV('lsrp:spot:timestamp_begin') failed ({e})")
                self.pv_tsdas = None
            try:
                self.pv_tsdae = epics.PV('lsrp:spot:timestamp_end')
            except Exception as e:
                print(f"WARNING: epics.PV('lsrp:spot:timestamp_end') failed ({e})")
                self.pv_tsdae = None
            try:
                self.pv_I = epics.PV('lsrp:VEPP4:CurrentE1-RB')
            except Exception as e:
                print(f"WARNING: epics.PV('lsrp:VEPP4:CurrentE1-RB') failed ({e})")
                self.pv_I = None

        # data file
        self.data_file = None
        self.data_filepath = None
        
        # current timestamp
        self.current_ts_start_step = None
        self.current_ts_end_step   = None

    #                 drawing
    # change scanning zone for the new parameters
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

    # draw detector
    def draw_detector(self, cx, cy):
        canvas_cx = self.scan_x0 + cx
        canvas_cy = self.scan_y0 + cy
        hw = DETECTOR_WIDTH / 2
        hh = DETECTOR_HEIGHT / 2
        self.canvas.coords(self.detector_item,
                           canvas_cx - hw, canvas_cy - hh,
                           canvas_cx + hw, canvas_cy + hh)
        x_steps = (cx - self.scan_w / 2) / MP
        y_steps = (cy - self.scan_h / 2) / MP
        self.posx_label.config(text=f"x={x_steps:.0f}")
        self.posy_label.config(text=f"y={y_steps:.0f}")

    #                         move detector
    def move_support(self, cx, cy):
        if self.dev is None:
            return time.time()

        dx = (cx - self.scan_w / 2) / MP
        dy = (cy - self.scan_h / 2) / MP
        sc = self.motor_scale
        tx = self.ref_x + int(round(dx * SCALE[0] / sc[0]))
        ty = self.ref_y - int(round(dy * SCALE[1] / sc[1]))
        t_move = time.time()
        try:
            self.dev.move_to(tx, ty)
        except Exception as e:
            self.status_var.set(f"Hardware error: {e}")
            return None
        return t_move

    #                    reads data from GEM and stores it
    def save_data(self, time_idx, ts_start_step, ts_end_step):
        if self.data_file is None:
            return False

        # reading data
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
                    arr = arr[::-1, :]            # flip y: y=0 сверху
                    data_flat = arr.flatten().astype(int).tolist()
                except Exception:
                    data_flat = [0] * (PADS_X * PADS_Y)
                    arr = None

        # reading timestamps for data
        ts_start_data = epics_ts_to_str(self.pv_tsdas.get() if self.pv_tsdas else None)
        ts_end_data   = epics_ts_to_str(self.pv_tsdae.get() if self.pv_tsdae else None)
        
        # works if ts_end_data > ts_end_step
        try:
            ts_end_step_dt = datetime.datetime.fromisoformat(ts_end_step)
            ts_end_data_dt = datetime.datetime.fromisoformat(ts_end_data) if ts_end_data else None
        except (ValueError, TypeError):
            ts_end_data_dt = None

        if ts_end_data_dt is None or ts_end_data_dt <= ts_end_step_dt:
            return False

        # draws data map (lower canvas)
        if arr is not None:
            draw_heatmap(heat_canvas, arr)

        c_x = int(round((self.cur_x - self.scan_w / 2) / MP)) + DETECTOR_WIDTH // 2 // MP
        c_y = int(round((self.cur_y - self.scan_h / 2) / MP)) + DETECTOR_HEIGHT // 2 // MP

        I = self.pv_I.get()

        # saves data to file
        data_str = " ".join(str(v) for v in data_flat)
        line = (f"{ts_start_step} {ts_end_step} "
                f"{ts_start_data} {ts_end_data} "
                f"{I:.6f} {c_x} {c_y} {data_str}\n")
        self.data_file.write(line)
        self.data_file.flush()
        return True

    #             builds path for the moving (three different options right now)
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

    #                             buttons
    def start(self):
        # start or continue
        if self.running or self.returning or self.waiting:
            return
        if self.paused:
            self.paused = False
            self.running = True
            self.status_var.set("Running (retry)")
            self.step()
            return

        # create data file before starting and close previous
        if self.data_file is not None:
            self.data_file.close()
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.data_filepath = f"data/data{ts}.txt"
        self.data_file = open(self.data_filepath, "w")
        print(f"Writing to {self.data_filepath}")

        # starting parameters
        self.path = self.build_path()
        self.position = 0
        self.step_times = []
        self.last_step_time = None
        self.total_time = 0.0
        self.remaining_s = 0.0
        
        # update tss
        self.current_ts_start_step = None
        self.current_ts_end_step   = None

        # moving to the start
        self.waiting = True
        self.status_var.set("Wait")
        self._move_to_start()

    # moving to the start function
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

    # pausing
    def pause(self):
        if not self.running:
            return
        self.running = False
        self.paused = True
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
            
        # update tss
        self.current_ts_start_step = None
        self.current_ts_end_step   = None
        self.status_var.set("Paused")

    # stop scanning and return to the center
    def stop(self):
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.paused = False
        self.waiting = False
        self.returning = True
        self.status_var.set("Returning to center...")
        self.move_support(*self.center_position())
        self._return_step()

    #                             moving detector by steps
    def step(self):
        # check ending
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
            if self.data_file is not None:
                self.data_file.close()
                print(f"Closed {self.data_filepath}")
                self.data_file = None
            return

        # counting timestamps (фиксируем ts_end_step один раз на позицию,
        # чтобы данные успели догнать его и не ждать бесконечно)
        if self.current_ts_end_step is None:
            now = time.time()
            if self.last_step_time is not None and len(self.step_times) < 10:
                self.step_times.append(now - self.last_step_time)
            self.last_step_time = now

            self.current_ts_start_step = fmt_ts(now - self.t_var.get())
            self.current_ts_end_step   = fmt_ts(now)

        ts_start_step = self.current_ts_start_step
        ts_end_step   = self.current_ts_end_step

        # saves data
        ok = self.save_data(self.position, ts_start_step, ts_end_step)
        if not ok:
            self.status_var.set(f"Waiting for fresh data (pos {self.position})")
            delay_ms = max(10, int(self.t_var.get() * 1000))
            self.after_id = self.root.after(delay_ms, self.step)
            return

        # данные свежие -- сбрасываем зафиксированные ts для следующей позиции
        self.current_ts_start_step = None
        self.current_ts_end_step   = None

        # motors moving
        self.cur_x, self.cur_y = self.path[self.position]
        self.draw_detector(self.cur_x, self.cur_y)
        self.root.update_idletasks()

        t_move = self.move_support(self.cur_x, self.cur_y)
        if t_move is None:
            self.running = False
            self.paused = True
            return

        self.position += 1
        self.status_var.set(f"Running {self.position}/{len(self.path)}")

        # remaining time counting
        if len(self.step_times) >= 10:
            avg_step = sum(self.step_times) / len(self.step_times)
            remaining_steps = len(self.path) - self.position
            self.remaining_s = avg_step * remaining_steps
            self.time_label.config(text=self._format_time(self.remaining_s))
        elif len(self.step_times) > 0:
            self.time_label.config(text="00:00.000 (measuring...)")

        delay_ms = max(1, int(self.t_var.get() * 1000))
        self.after_id = self.root.after(delay_ms, self.step)

    # formating time from seconds
    @staticmethod
    def _format_time(seconds):
        total = int(seconds * 1000)
        mm = total // 60000
        ss = (total % 60000) // 1000
        ms = total % 1000
        return f"{mm:02d}:{ss:02d}.{ms:03d}"

    #                      move to center
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

        delay_ms = 10
        self.after_id = self.root.after(delay_ms, self._return_step)


#                           data drawing (lower canvas)
def draw_heatmap(canvas, data_2d):
    """Draw 2D array as a heatmap filling the whole canvas.
    Flip x so that the map corresponds to the detector orientation."""
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
            b = int(255 * (v))
            color = f"#{b:02x}{b:02x}{b:02x}"
            x0 = (cols - 1 - j) * cell_w    # flip x для heatmap
            y0 = (rows - 1 - i) * cell_h
            canvas.create_rectangle(x0, y0, x0 + cell_w, y0 + cell_h,
                                    fill=color, outline="", tags="heat")


#                           frame parameters
root = Tk()
root.title("Efficiency test")
root.geometry("920x760")
root.resizable(False, False)

x_var      = IntVar(value=32)
y_var      = IntVar(value=20)
xstep_var  = IntVar(value=1)
ystep_var  = IntVar(value=1)
t_var      = DoubleVar(value=0.01)
cycles_var = IntVar(value=1)
modes      = ["one side", "both sides"]
modes_var  = StringVar(value=modes[0])
status_var = StringVar(value="Not ready")

#                                   canvases
pos_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
pos_canvas.place(x=10, y=10)

scan_item     = pos_canvas.create_rectangle(0, 0, 1, 1, fill="blue")
green_item    = pos_canvas.create_rectangle(0, 0, 1, 1, fill="green")
detector_item = pos_canvas.create_rectangle(0, 0, 1, 1, fill="grey")

heat_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
heat_canvas.place(x=10, y=320)

heat_data = [[0.0] * PADS_X for _ in range(PADS_Y)]
draw_heatmap(heat_canvas, heat_data)

#                                   controls
COL_X_LABEL = 430
COL_X_ENTRY = 460
COL_X_POS   = 620
COL_W_ENTRY = 60

ttk.Label(root, text="size (motor steps)").place(x=COL_X_LABEL, y=10)

ttk.Label(root, text="x").place(x=COL_X_LABEL, y=50)
ttk.Entry(textvariable=x_var, width=8).place(x=COL_X_ENTRY, y=50)
posx_label = ttk.Label(root, text="x=0")
posx_label.place(x=COL_X_POS, y=50)

ttk.Label(root, text="y").place(x=COL_X_LABEL, y=90)
ttk.Entry(textvariable=y_var, width=8).place(x=COL_X_ENTRY, y=90)
posy_label = ttk.Label(root, text="y=0")
posy_label.place(x=COL_X_POS, y=90)

ttk.Label(root, text="step (motor steps)").place(x=COL_X_LABEL, y=140)

ttk.Label(root, text="x").place(x=COL_X_LABEL, y=180)
ttk.Entry(textvariable=xstep_var, width=8).place(x=COL_X_ENTRY, y=180)

ttk.Label(root, text="y").place(x=COL_X_LABEL, y=220)
ttk.Entry(textvariable=ystep_var, width=8).place(x=COL_X_ENTRY, y=220)

ttk.Label(root, text="t (s)").place(x=COL_X_LABEL, y=260)
ttk.Entry(textvariable=t_var, width=8).place(x=COL_X_ENTRY, y=260)

ttk.Label(root, text="time:").place(x=COL_X_LABEL, y=300)
time_label = ttk.Label(root, text="00:00.000")
time_label.place(x=COL_X_ENTRY + 20, y=300)

BTN_X = COL_X_LABEL
ttk.Button(root, text="start").place(x=BTN_X, y=360)
ttk.Button(root, text="pause").place(x=BTN_X, y=400)
ttk.Button(root, text="stop").place(x=BTN_X, y=440)

BLOCK_Y = 520

ttk.Label(root, text="mode").place(x=COL_X_LABEL, y=BLOCK_Y)
ttk.Combobox(textvariable=modes_var, values=modes,
             state="readonly", width=14).place(x=COL_X_ENTRY + 30, y=BLOCK_Y)

ttk.Label(root, text="cycles").place(x=COL_X_LABEL, y=BLOCK_Y + 40)
ttk.Entry(textvariable=cycles_var, width=8).place(x=COL_X_ENTRY + 30, y=BLOCK_Y + 40)

ttk.Label(root, text="status").place(x=COL_X_LABEL, y=BLOCK_Y + 80)
ttk.Entry(textvariable=status_var, state="readonly", width=24).place(x=COL_X_ENTRY + 30, y=BLOCK_Y + 80)

#                                           connect to devices
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

scanner = Scanner(root, pos_canvas, detector_item, green_item, scan_item,
                  x_var, y_var, xstep_var, ystep_var, t_var, cycles_var,
                  status_var, modes_var, posx_label, posy_label, time_label, dev=dev)

#                                       check actions
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

#                                       program start
root.mainloop()
