#!/usr/bin/env python3
"""
start: starts or continiues scanning
pause: pause scanning with the current position
stop:  stops scanning and returns detector to the default position (center in the top corner of the scanning zone)

zones
blue: scanning zone
grey: detector center 
green: laser beam
"""

from tkinter import *
from tkinter import ttk
import math

CAN_WIDTH       = 370
CAN_HEIGHT      = 300
MP = 4

DETECTOR_WIDTH  = 32 * MP
DETECTOR_HEIGHT = 20 * MP
GREEN_WIDTH  = 20 * MP
GREEN_HEIGHT = 10 * MP

class Scanner:
    def __init__(self, root, canvas, detector_item, green_item, scan_item,
                 x_var, y_var, xstep_var, ystep_var, t_var, cycles_var, status_var, modes_var):
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

        # statuses and detector characteristics
        self.running  = False
        self.paused   = False
        self.returning = False
        self.position = 0
        self.path = []
        self.after_id = None

        # current detector position
        self.cur_x = 0.0
        self.cur_y = 0.0

        self.scan_x0 = 0.0
        self.scan_y0 = 0.0
        self.scan_w  = 0.0
        self.scan_h  = 0.0

    def update_geometry(self):
        # scanning zone
        w = max(self.x_var.get() * MP, 10)
        h = max(self.y_var.get() * MP, 10)
        self.scan_w = w
        self.scan_h = h
        self.scan_x0 = (CAN_WIDTH - w) / 2
        self.scan_y0 = (CAN_HEIGHT - h) / 2
        self.canvas.coords(self.scan_item,
                           self.scan_x0, self.scan_y0,
                           self.scan_x0 + w, self.scan_y0 + h)

        # laser beam
        gx = self.scan_x0 + (w - GREEN_WIDTH) / 2
        gy = self.scan_y0 + (h - GREEN_HEIGHT) / 2
        self.canvas.coords(self.green_item,
                           gx, gy, gx + GREEN_WIDTH, gy + GREEN_HEIGHT)

        # detector
        self.draw_detector(self.cur_x, self.cur_y)

    def draw_detector(self, cx, cy):
        canvas_cx = self.scan_x0 + cx
        canvas_cy = self.scan_y0 + cy
        hw = DETECTOR_WIDTH / 2
        hh = DETECTOR_HEIGHT / 2
        self.canvas.coords(self.detector_item,
                           canvas_cx - hw, canvas_cy - hh,
                           canvas_cx + hw, canvas_cy + hh)

    # steps
    def build_path(self):
        xs = max(self.xstep_var.get() * MP, 1)
        ys = max(self.ystep_var.get() * MP, 1)
        
        nx = int(self.scan_w // xs) + 1
        ny = int(self.scan_h // ys) + 1
        lastx = 0
        cycles_num = cycles_var.get()
        path = []
        
        for n in range(cycles_num):
            for j in range(ny):
                cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
                for i in cols:
                    path.append((i * xs, j * ys))
                    lastx = i*xs
            if(self.modes_var.get() == "steps both sides"):
                for j in range(ny-1, -1, -1):
                    if lastx == 0:
                        cols = range(nx) if j % 2 == 0 else range(nx - 1, -1, -1)
                    else:
                        cols = range(nx) if j % 2 != 0 else range(nx - 1, -1, -1)
                    for i in cols:
                        path.append((i * xs, j * ys))
        return path

    def start_position(self):
        return (0.0, 0.0)

    def start(self):
        if self.running or self.returning:
            return
        if self.paused:
            self.paused = False
            self.running = True
            self.status_var.set("Running")
            self.step()
            return
        # new scan from the current position
        self.cur_x, self.cur_y = self.start_position()
        self.draw_detector(self.cur_x, self.cur_y)
        self.path = self.build_path()
        self.position = 0
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
        if self.after_id is not None:
            self.root.after_cancel(self.after_id)
            self.after_id = None
        self.running = False
        self.paused = False
        self.returning = True
        self.status_var.set("Returning...")
        self._return_step()

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
        self.position += 1
        self.status_var.set(f"Running {self.position}/{len(self.path)}")

        delay_ms = max(1, int(self.t_var.get() * 1000))
        self.after_id = self.root.after(delay_ms, self.step)

    # return after stop
    def _return_step(self):
        if not self.returning:
            return
        tx, ty = self.start_position()
        dx = tx - self.cur_x
        dy = ty - self.cur_y
        dist = math.hypot(dx, dy)

        # one step per t, but not more than corners
        step_len = MP # speed of returning
        if dist <= step_len:
            self.cur_x, self.cur_y = tx, ty
            self.draw_detector(self.cur_x, self.cur_y)
            self.returning = False
            self.status_var.set("Stopped")
            return

        self.cur_x += dx / dist * step_len
        self.cur_y += dy / dist * step_len
        self.draw_detector(self.cur_x, self.cur_y)
        self.status_var.set(f"Returning... ({dist:.1f} px)")

        delay_ms = max(1, int(self.t_var.get() * 1000))
        self.after_id = self.root.after(delay_ms, self._return_step)


root = Tk()
root.title("Efficiency test")
root.geometry("640x480")
root.resizable(False, False)

# app variables
x_var      = IntVar(value=52)
y_var      = IntVar(value=30)
xstep_var  = IntVar(value=1)
ystep_var  = IntVar(value=1)
t_var      = DoubleVar(value=0.01)
cycles_var = IntVar(value=1)
modes      = ["steps one side", "steps both sides"]
modes_var  = StringVar(value=modes[0])
status_var = StringVar(value="Not ready")

ttk.Label(root, text="size").place(x=510, y=10)
ttk.Label(root, text="x").place(x=405, y=50)
ttk.Entry(textvariable=x_var).place(x=420, y=50)
ttk.Label(root, text="y").place(x=405, y=90)
ttk.Entry(textvariable=y_var).place(x=420, y=90)

ttk.Label(root, text="step").place(x=510, y=130)
ttk.Label(root, text="x").place(x=405, y=170)
ttk.Entry(textvariable=xstep_var).place(x=420, y=170)
ttk.Label(root, text="y").place(x=405, y=210)
ttk.Entry(textvariable=ystep_var).place(x=420, y=210)

ttk.Label(root, text="t (s)").place(x=405, y=250)
ttk.Entry(textvariable=t_var).place(x=420, y=250)

pos_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
pos_canvas.place(x=10, y=10)

scan_item     = pos_canvas.create_rectangle(0, 0, 1, 1, fill="blue")
green_item    = pos_canvas.create_rectangle(0, 0, 1, 1, fill="green")
detector_item = pos_canvas.create_rectangle(0, 0, 1, 1, fill="grey")

scanner = Scanner(root, pos_canvas, detector_item, green_item, scan_item,
                  x_var, y_var, xstep_var, ystep_var, t_var, cycles_var, status_var, modes_var)
scanner.update_geometry()
scanner.draw_detector(*scanner.start_position())

# trace geometry
x_var.trace_add("write", lambda *_: scanner.update_geometry())
y_var.trace_add("write", lambda *_: scanner.update_geometry())

ttk.Button(root, text="start", command=scanner.start).place(x=470, y=300)
ttk.Button(root, text="pause", command=scanner.pause).place(x=470, y=350)
ttk.Button(root, text="stop",  command=scanner.stop ).place(x=470, y=400)

ttk.Label(root, text="mode").place(x=230, y=360)
ttk.Combobox(textvariable=modes_var, values=modes, state="readonly").place(x=10, y=360)

ttk.Label(root, text="cycles").place(x=220, y=395)
ttk.Entry(textvariable=cycles_var).place(x=10, y=395)

ttk.Entry(textvariable=status_var, state="readonly").place(x=10, y=430)

root.mainloop()
