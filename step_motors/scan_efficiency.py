#!/usr/bin/env python3
import devices, time, sys
import numpy as np
import tkinter as tk

# scan zone, mm
x_zone = 120.
y_zone = 50.
# steps in mm
x_step = 4.
y_step = 4.
SCALE = [0.0125, 0.01]


wait_time = 0.001
n_cycles = 1


def config_grid(x_med, y_med, x_zone, y_zone, x_step, y_step):
	x_grid = np.linspace(x_med - x_zone / SCALE[0] / 2, x_med + x_zone / SCALE[0] / 2,
						int(x_zone / x_step) + 1)
	y_grid = np.linspace(y_med - y_zone / SCALE[1] / 2, y_med + y_zone / SCALE[1] / 2,
						int(y_zone / y_step) + 1)
	n_steps = (int(x_zone/x_step) + 1) * (int(y_zone/y_step) + 1)
	print(x_grid, y_grid, n_steps) 
	return x_grid, y_grid, n_steps

	
def main():
	# central point
#	x_med = 9000
#	y_med = 17300
	dev = devices.Support("Support")
	x_med = dev.status["x"]
	y_med = dev.status["y"]

	x_grid, y_grid, n_steps = config_grid(x_med, y_med, x_zone, y_zone, x_step, y_step)
	print("Total steps per (forward-backward) cycle:", n_steps*2)
	t = n_steps*(wait_time+0.8)*2
	print("Total time per cycle:",  t, "sec.")
	print("Total time for", n_cycles, "cycles:",  int(t*n_cycles) // 60, "min.",
													int(t*n_cycles) % 60, "sec.")
	if "n" in input("Procceed? ('no' to cancel) "):
		sys.exit()

	if "-c" in sys.argv[0:]:
		print("Calibrating")
		dev.calibrate()

	print("Moving to the initial point")
	dev.move_to(x_grid[0], y_grid[0])
	# save data while moving
	direction = 1
	cycle = 1
	input("Press <Enter> to start scan.")
	try:
		for cycle in range(int(n_cycles)):
			for direction in [0,1]:
				if direction==0:
					print(F"Cycle #{cycle}, forward direction.")
				else:
					print(F"Cycle #{cycle}, backward direction.")
				n = 0
				for y in y_grid:
					for x in x_grid:
						dev.move_to(x, y)
						n += 1
						print(F"Point {n} of {int(n_steps)} (cycle {cycle+0.5*direction} of {int(n_cycles)}): x = {x:5.0f}, y = {y:5.0f}")
						time.sleep(wait_time)
						# save data here
					x_grid = np.flip(x_grid)
				y_grid = np.flip(y_grid)
	except KeyboardInterrupt:
		pass
	
	print("Moving to the lateral point")
	dev.move_to(x_med + x_zone / SCALE[0], y_med)

	input("Press <Enter> to move to central point.")
	print("Moving to the central point")
	dev.move_to(x_med, y_med)
	
	
		
if __name__ == "__main__":
	main()


