#!/usr/bin/env python3
"""GUI-обёртка над ScannerCore."""

from tkinter import *
from tkinter import ttk
import os
import time
import numpy as np

from scanner_core import ScannerCore, MP, PADS_X, PADS_Y, DETECTOR_WIDTH, DETECTOR_HEIGHT

CAN_WIDTH  = 370
CAN_HEIGHT = 300
GREEN_WIDTH  = 20 * MP
GREEN_HEIGHT = 10 * MP
HEAT_COLS = PADS_X
HEAT_ROWS = PADS_Y

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# ----------------------------- heatmap -----------------------------
def draw_heatmap(canvas, data_2d):
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
            x0 = (cols - 1 - j) * cell_w    # flip x
            y0 = i * cell_h
            canvas.create_rectangle(x0, y0, x0 + cell_w, y0 + cell_h,
                                    fill=color, outline="", tags="heat")

# ----------------------------- GUI -----------------------------
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

# canvas
pos_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
pos_canvas.place(x=10, y=10)
scan_item     = pos_canvas.create_rectangle(0, 0, 1, 1, fill="blue")
green_item    = pos_canvas.create_rectangle(0, 0, 1, 1, fill="green")
detector_item = pos_canvas.create_rectangle(0, 0, 1, 1, fill="grey")

heat_canvas = Canvas(bg="white", width=CAN_WIDTH, height=CAN_HEIGHT)
heat_canvas.place(x=10, y=320)
heat_data = [[0.0] * HEAT_COLS for _ in range(HEAT_ROWS)]
draw_heatmap(heat_canvas, heat_data)

# controls
COL_X_LABEL = 430
COL_X_ENTRY = 460
COL_X_POS   = 620

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

ttk.Button(root, text="start").place(x=COL_X_LABEL, y=360)
ttk.Button(root, text="pause").place(x=COL_X_LABEL, y=400)
ttk.Button(root, text="stop").place(x=COL_X_LABEL, y=440)

BLOCK_Y = 520
ttk.Label(root, text="mode").place(x=COL_X_LABEL, y=BLOCK_Y)
ttk.Combobox(textvariable=modes_var, values=modes,
             state="readonly", width=14).place(x=COL_X_ENTRY + 30, y=BLOCK_Y)
ttk.Label(root, text="cycles").place(x=COL_X_LABEL, y=BLOCK_Y + 40)
ttk.Entry(textvariable=cycles_var, width=8).place(x=COL_X_ENTRY + 30, y=BLOCK_Y + 40)
ttk.Label(root, text="status").place(x=COL_X_LABEL, y=BLOCK_Y + 80)
ttk.Entry(textvariable=status_var, state="readonly", width=24).place(x=COL_X_ENTRY + 30, y=BLOCK_Y + 80)

# scanner core
scanner = ScannerCore(
    x_zone=x_var.get(), y_zone=y_var.get(),
    x_step=xstep_var.get(), y_step=ystep_var.get(),
    t_var=t_var.get(), cycles=cycles_var.get(),
    both_sides=(modes_var.get() == "both sides"),
    dev=None,   # ← сюда devices.Support, если есть
)

# callbacks
def on_status(text):
    status_var.set(text)

def on_data(arr):
    draw_heatmap(heat_canvas, arr)

def on_step(pos, total, waiting):
    status_var.set(f"Running {pos}/{total}" + (" (waiting)" if waiting else ""))

def on_finished():
    status_var.set("Finished")

def on_error(text):
    status_var.set(f"Error: {text}")

scanner.on_status = on_status
scanner.on_data = on_data
scanner.on_step = on_step
scanner.on_finished = on_finished
scanner.on_error = on_error

# geometry
def update_geometry():
    w = max(x_var.get() * MP, 10)
    h = max(y_var.get() * MP, 10)
    scan_x0 = (CAN_WIDTH - w) / 2
    scan_y0 = (CAN_HEIGHT - h) / 2
    pos_canvas.coords(scan_item, scan_x0, scan_y0, scan_x0 + w, scan_y0 + h)
    gx = scan_x0 + (w - GREEN_WIDTH) / 2
    gy = scan_y0 + (h - GREEN_HEIGHT) / 2
    pos_canvas.coords(green_item, gx, gy, gx + GREEN_WIDTH, gy + GREEN_HEIGHT)

    # detector
    cx = scanner.cur_x
    cy = scanner.cur_y
    canvas_cx = scan_x0 + cx
    canvas_cy = scan_y0 + cy
    hw = DETECTOR_WIDTH / 2
    hh = DETECTOR_HEIGHT / 2
    pos_canvas.coords(detector_item,
                      canvas_cx - hw, canvas_cy - hh,
                      canvas_cx + hw, canvas_cy + hh)
    posx_label.config(text=f"x={(cx - w/2)/MP:.0f}")
    posy_label.config(text=f"y={(cy - h/2)/MP:.0f}")

x_var.trace_add("write", lambda *_: update_geometry())
y_var.trace_add("write", lambda *_: update_geometry())
update_geometry()

# buttons
def on_start():
    scanner.x_zone = x_var.get()
    scanner.y_zone = y_var.get()
    scanner.x_step = xstep_var.get()
    scanner.y_step = ystep_var.get()
    scanner.t_var = t_var.get()
    scanner.cycles = cycles_var.get()
    scanner.both_sides = (modes_var.get() == "both sides")
    scanner.scan_w = scanner.x_zone * MP
    scanner.scan_h = scanner.y_zone * MP
    scanner.start()
    _poll()

def on_pause():
    scanner.pause()

def on_stop():
    scanner.stop()

def _poll():
    if scanner.running:
        scanner.poll()
        update_geometry()
        root.after(max(10, int(t_var.get() * 1000)), _poll)

for child in root.winfo_children():
    if isinstance(child, ttk.Button):
        txt = child.cget("text")
        if txt == "start":
            child.config(command=on_start)
        elif txt == "pause":
            child.config(command=on_pause)
        elif txt == "stop":
            child.config(command=on_stop)

root.mainloop()
