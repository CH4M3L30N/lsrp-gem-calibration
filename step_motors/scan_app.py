#!/usr/bin/env python3
from tkinter import *
from tkinter import ttk
import time

def on_click():
    if start_button["text"] == "start":    
        finish = False
        status.set("Started")
        start_button["text"] = "stop"
        start_button["state"] = "enabled"
        for n in range(cycles.get()):
            for j in range(y.get() // ystep.get() + 1):
                for i in range(x.get() // xstep.get() + 1):
                    if j % 2 == 0:
                        newx = (det_posx + i * xstep.get())
                    else:
                        newx = (rect_posx + x.get() - detector_width/2 - i * xstep.get())
                    newy = det_posy + j * ystep.get()
                    pos_canvas.coords(detector, newx, newy, newx + detector_width, newy + detector_height)
                    root.update_idletasks()
                    time.sleep(t.get())
                    if start_button["text"] == "start":
                        finish = True
                        break
                        
            for j in range(y.get() // ystep.get(), -1, -1):
                for i in range(x.get() // xstep.get() + 1):
                    if j % 2 != 0:
                        newx = (det_posx + i * xstep.get())
                    else:
                        newx = (rect_posx + x.get() - detector_width/2 - i * xstep.get())
                    newy = det_posy + j * ystep.get()
                    pos_canvas.coords(detector, newx, newy, newx + detector_width, newy + detector_height)
                    root.update_idletasks()
                    time.sleep(t.get())
                    if start_button["text"] == "start":
                        finish = True
                        break
        
        status.set("Finished")
    else:
        start_button["text"] = "start"

# 1. Create the main window
root = Tk()
root.title("Efficiency test")
root.geometry("640x480")
root.resizable(False, False)

# Create variables
x = IntVar(value = 250)
y = IntVar(value = 80)

xstep = IntVar(value = 8)
ystep = IntVar(value = 8)
t = DoubleVar(value = 0.01)

cycles = IntVar(value = 1)

modes = ["steps", "continuous"]
modes_var = StringVar(value=modes[0])

status = StringVar(value = "Not ready")

can_width = 320
can_height = 240
detector_width = 40
detector_height = 20
rect_posx = (can_width - x.get())/2
rect_posy = (can_height - y.get())/2
det_posx = rect_posx - detector_width/2
det_posy = rect_posy - detector_height/2

# 2. Add widgets

# right part
size_label = ttk.Label(root, text="size")
size_label.place(x = 510, y = 10)

#entry.get()
x_label = ttk.Label(root, text="x")
x_label.place(x = 405, y = 50)
x_entry = ttk.Entry(textvariable=x)
#x_entry.insert(0, "250")
x_entry.place(x = 420, y = 50)

y_label = ttk.Label(root, text="y")
y_label.place(x = 405, y = 90)
y_entry = ttk.Entry(textvariable=y)
#y_entry.insert(0, "80")
y_entry.place(x = 420, y = 90)

step_label = ttk.Label(root, text="step")
step_label.place(x = 510, y = 130)

xstep_label = ttk.Label(root, text="x")
xstep_label.place(x = 405, y = 170)
xstep_entry = ttk.Entry(textvariable=xstep)
#xstep_entry.insert(0, "8")
xstep_entry.place(x = 420, y = 170)

ystep_label = ttk.Label(root, text="y")
ystep_label.place(x = 405, y = 210)
ystep_entry = ttk.Entry(textvariable=ystep)
#ystep_entry.insert(0, "8")
ystep_entry.place(x = 420, y = 210)

t_label = ttk.Label(root, text="t")
t_label.place(x = 405, y = 250)
t_entry = ttk.Entry(textvariable=t)
#t_entry.insert(0, "0.01")
t_entry.place(x = 420, y = 250)

start_button = ttk.Button(root, text="start", command=on_click)
start_button.place(x = 470, y = 300)

pause_button = ttk.Button(root, text="pause", command=on_click)
pause_button.place(x = 470, y = 350)

# left part
pos_canvas = Canvas(bg="white", width=can_width, height=can_height)
pos_canvas.place(x = 10, y = 10)
rect = pos_canvas.create_rectangle(rect_posx, rect_posy, rect_posx + x.get(), rect_posy + y.get(), fill = "blue")
detector = pos_canvas.create_rectangle(det_posx, det_posy, det_posx + detector_width, det_posy + detector_height, fill = "grey")

mode_label = ttk.Label(root, text="mode")
mode_label.place(x = 230, y = 360)
mode_box = ttk.Combobox(textvariable = modes_var, values = modes, state="readonly")
mode_box.place(x = 10, y = 360)

cycles_label = ttk.Label(root, text="cycles")
cycles_label.place(x = 220, y = 395)
cycles_entry = ttk.Entry(textvariable=cycles)
cycles_entry.place(x = 10, y = 395)

status_entry = ttk.Entry(textvariable = status, state="readonly")
status_entry.place(x = 10, y = 430)

# 3. Start the event loop
root.mainloop()
