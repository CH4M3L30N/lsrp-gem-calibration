#!/usr/bin/env python3
"""
Combined scanner: GUI animation + stepper-motor control.

Buttons:
  start     -- starts / continues scanning
  pause     -- pauses scanning (current position kept)
  stop      -- stops and returns detector to the top-left corner of the scan zone
  calibrate -- runs dev.calibrate() (no-op in demo mode)

Zones:
  blue  -- scanning zone (size = x_var x y_var motor steps, drawn as x_var*MP x y_var*MP px)
  grey  -- detector (center = laser beam position)
  green -- laser beam

x_var, y_var, xstep_var, ystep_var are in MOTOR STEPS.
SCALE = [mm/step_x, mm/step_y] -- used only to build the motor grid in mm,
                                  not for GUI conversion (GUI uses MP).
"""

from tkinter import *
from tkinter import ttk
import math
import time

# ----------------------------- optional hardware -----------------------------
try:
    import devices
    HAVE_DEVICES = True
except ImportError:
    HAVE_DEVICES = False
    print("WARNING: 'devices' not found -- DEMO mode (no motors)")

# ----------------------------- constants -----------------------------
CAN_WIDTH       = 370
CAN_HEIGHT      = 300
MP              = 4           # GUI px per motor step (drawing only)

DETECTOR_WIDTH  = 32 * MP
DETECTOR_HEIGHT = 20 * MP
GREEN_WIDTH     = 20 * MP
GREEN_HEIGHT    = 10 * MP

# mm per motor step (from scan_efficiency.py)
SCALE = [1, 2]

# ----------------------------- Scanner -----------------------------
class Scanner:
    def __init__(self, root, canvas, detector_item, green_item, scan_item,
                 x_var, y_var, xstep_var, ystep_var, t_var, cycles_var,
                 status_var, modes_var, dev=None):
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
        self.dev = dev

        # state
        self.running   = False
        self.paused    = False
        self.returning = False
        self.position  = 0
        self.path      = []
        self.after_id  = None

        # current detector center (GUI px within blue zone)
        self.cur_x = 0.0
        self.cur_y = 0.0

        # geometry (GUI px)
        self.scan_x0 = 0.0
        self.scan_y0 = 0.0
        self.scan_w  = 0.0
        self.scan_h  = 0.0

        # physical grid (motor steps)
        self.x_grid = None
        self.y_grid = None
        self.x_med  = None
        self.y_med  = None

    # ---------- physical grid ----------
    def config_grid(self):
        """Build motor-step grid. x_var/y_var are in motor steps."""
        x_zone = max(self.x_var.get(), 1)      # motor steps
        y_zone = max(self.y_var.get(), 1)
        x_step = max(self.xstep_var.get(), 1)
        y_step = max(self.ystep_var.get(), 1)

        if self.dev is not None:
            self.x_med = self.dev.status["x"]
            self.y_med = self.dev.status["y"]
        else:
            self.x_med = 0
            self.y_med = 0

        nx = int(x_zone / x_step) + 1
        ny = int(y_zone / y_step) + 1
        x0 = self.x_med - x_zone / 2
        y0 = self.y_med - y_zone / 2
        self.x_grid = [x0 + i * x_step for i in range(nx)]
        self.y_grid = [y0 + j * y_step for j in range(ny)]
        return self.x_grid, self.y_grid

    def gui_to_phys(self, cx, cy):
        """GUI px (within blue zone) -> motor steps."""
        if self.x_grid is None or self.y_grid is None:
            return (self.x_med or 0), (self.y_med or 0)
        x_phys = self.x_grid[0] + (cx / max(self.scan_w, 1)) * (self.x_grid[-1] - self.x_grid[0])
        y_phys = self.y_grid[0] + (cy / max(self.scan_h, 1)) * (self.y_grid[-1] - self.y_grid[0])
        return x_phys, y_phys

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

    # ---------- path ----------
    def build_path(self):
        """Path in GUI px (detector center positions within blue zone)."""
        xs = max(self.xstep_var.get() * MP, 1)
        ys = max(self.ystep_var.get() * MP, 1)
        nx = int(self.scan_w // xs) + 1
        ny = int(self.scan_h // ys) + 1
        n_cycles = max(self.cycles_var.get(), 1)
        both_sides = (self.modes_var.get() == "both sides")

        path = []
        for _ in range(n_cycles):
            # forward: snake top -> bottom
            for j in range(ny):
                cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
                for i in cols:
                    path.append((i * xs, j * ys))
            # backward: snake bottom -> top
            if both_sides:
                for j in range(ny - 1, -1, -1):
                    # continue snake pattern from last row
                    if (ny - 1) % 2 == 0:
                        # last forward row went left->right; backward starts right->left
                        cols = range(nx - 1, -1, -1) if j % 2 != 0 else range(nx)
                    else:
                        cols = range(nx) if j % 2 != 0 else range(nx - 1, -1, -1)
                    for i in cols:
                        path.append((i * xs, j * ys))
        return path

    def start_position(self):
        """Top-left corner of the scan zone (detector center)."""
        return (0.0, 0.0)

    # ---------- control ----------
    def calibrate(self):
        if self.dev is None:
            self.status_var.set("Calibrate: no device (demo)")
            return
        self.status_var.set("Calibrating...")
        self.root.update_idletasks()
        try:
            self.dev.calibrate()
            self.status_var.set("Calibrated")
        except Exception as e:
            self.status_var.set(f"Calibrate error: {e}")

    def start(self):
        if self.running or self.returning:
            return
        if self.paused:
            self.paused = False
            self.running = True
            self.status_var.set("Running")
            self.step()
            return

        # build grid and path
        self.config_grid()
        self.path = self.build_path()
        self.position = 0
        self.cur_x, self.cur_y = self.start_position()
        self.draw_detector(self.cur_x, self.cur_y)

        # move to first physical point (top-left)
        if self.dev is not None:
            x_phys, y_phys = self.gui_to_phys(*self.start_position())
            self.status_var.set("Moving to start...")
            self.root.update_idletasks()
            try:
                self.dev.move_to(x_phys, y_phys)
            except Exception as e:
                self.status_var.set(f"Move error: {e}")
                return

        self.running = True
        self.paused = False
        self.status_var.set("Running")
        self.step()

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
        """Stop and return to top-left corner of scan zone (shortest path)."""
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.paused = False
        self.returning = True
        self.status_var.set("Returning to start...")
        self._return_step()

    # ---------- scanning ----------
    def step(self):
        if not self.running:
            return
        if self.position >= len(self.path):
            self.running = False
            self.paused = False
            self.status_var.set("Finished")
            return

        self.cur_x, self.cur_y = self.path[self.position]
        self.draw_detector(self.cur_x, self.cur_y)

        # hardware move (skipped in demo)
        if self.dev is not None:
            x_phys, y_phys = self.gui_to_phys(self.cur_x, self.cur_y)
            try:
                self.dev.move_to(x_phys, y_phys)
            except Exception as e:
                self.status_var.set(f"Move error: {e}")
                self.running = False
                return

        self.position += 1
        self.status_var.set(f"Running {self.position}/{len(self.path)}")

        # save data here (placeholder)

        delay_ms = max(1, int(self.t_var.get() * 1000))
        self.after_id = self.root.after(delay_ms, self.step)

    # ---------- return to top-left ----------
    def _return_step(self):
        if not self.returning:
            return
        tx, ty = self.start_position()
        dx = tx - self.cur_x
        dy = ty - self.cur_y
        dist = math.hypot(dx, dy)

        step_len = MP
        if dist <= step_len:
            self.cur_x, self.cur_y = tx, ty
            self.draw_detector(self.cur_x, self.cur_y)
            # hardware: move to top-left physically
            if self.dev is not None and self.x_grid is not None:
                x_phys, y_phys = self.gui_to_phys(tx, ty)
                try:
                    self.dev.move_to(x_phys, y_phys)
                except Exception as e:
                    self.status_var.set(f"Move error: {e}")
                    self.returning = False
                    return
            self.returning = False
            self.status_var.set("Stopped (at start)")
            return

        self.cur_x += dx / dist * step_len
        self.cur_y += dy / dist * step_len
        self.draw_detector(self.cur_x, self.cur_y)
        self.status_var.set(f"Returning... ({dist:.1f} px)")

        delay_ms = max(1, int(self.t_var.get() * 1000))
        self.after_id = self.root.after(delay_ms, self._return_step)


# ----------------------------- main window -----------------------------
root = Tk()
root.title("Efficiency test")
root.geometry("640x520")
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

# ----------------------------- right panel -----------------------------
ttk.Label(root, text="size (motor steps)").place(x=480, y=10)
ttk.Label(root, text="x").place(x=405, y=50)
ttk.Entry(textvariable=x_var).place(x=420, y=50)
ttk.Label(root, text="y").place(x=405, y=90)
ttk.Entry(textvariable=y_var).place(x=420, y=90)

ttk.Label(root, text="step (motor steps)").place(x=480, y=130)
ttk.Label(root, text="x").place(x=405, y=170)
ttk.Entry(textvariable=xstep_var).place(x=420, y=170)
ttk.Label(root, text="y").place(x=405, y=210)
ttk.Entry(textvariable=ystep_var).place(x=420, y=210)

ttk.Label(root, text="t (s)").place(x=405, y=250)
ttk.Entry(textvariable=t_var).place(x=420, y=250)

# ----------------------------- canvas -----------------------------
pos_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
pos_canvas.place(x=10, y=10)

scan_item     = pos_canvas.create_rectangle(0, 0, 1, 1, fill="blue")
green_item    = pos_canvas.create_rectangle(0, 0, 1, 1, fill="green")
detector_item = pos_canvas.create_rectangle(0, 0, 1, 1, fill="grey")

# ----------------------------- hardware (optional) -----------------------------
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

# ----------------------------- scanner -----------------------------
scanner = Scanner(root, pos_canvas, detector_item, green_item, scan_item,
                  x_var, y_var, xstep_var, ystep_var, t_var, cycles_var,
                  status_var, modes_var, dev=dev)
scanner.update_geometry()
scanner.draw_detector(*scanner.start_position())

x_var.trace_add("write", lambda *_: scanner.update_geometry())
y_var.trace_add("write", lambda *_: scanner.update_geometry())

# ----------------------------- buttons (4 independent) -----------------------------
ttk.Button(root, text="start",     command=scanner.start).place(x=470, y=300)
ttk.Button(root, text="pause",     command=scanner.pause).place(x=470, y=340)
ttk.Button(root, text="stop",      command=scanner.stop).place(x=470, y=380)
ttk.Button(root, text="calibrate", command=scanner.calibrate).place(x=470, y=420)

# ----------------------------- bottom panel -----------------------------
ttk.Label(root, text="mode").place(x=230, y=360)
ttk.Combobox(textvariable=modes_var, values=modes, state="readonly").place(x=10, y=360)

ttk.Label(root, text="cycles").place(x=220, y=395)
ttk.Entry(textvariable=cycles_var).place(x=10, y=395)

ttk.Entry(textvariable=status_var, state="readonly").place(x=10, y=430)

# demo-mode hint
if dev is None:
    ttk.Label(root, text="(demo: no motors)",
              foreground="red").place(x=10, y=460)

# ----------------------------- run -----------------------------
root.mainloop()
