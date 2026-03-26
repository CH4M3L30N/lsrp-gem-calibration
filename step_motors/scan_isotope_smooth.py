#!/usr/bin/env python3
import time, sys
import numpy as np

from motor import Motor
from device_config import DeviceConfig

SCALE = [0.0125, 0.01]
# scan zone, steps (mm / scale)
x_zone = 110. / SCALE[0]
y_zone = 0.50 / SCALE[0]

x_timer = 138890
y_timer = 138890

x_step_time = 1e9*4e-7
y_step_time = 1e9*4e-7

n_cycles = 1

print(x_zone, y_zone)

#sys.exit()

# BUG: контроллер отвечает "3" на команду "идти с прецизионной скоростью",
# 	затем не выдаёт ответов на любые команды, но продолжает несколько актов
# 	движения в том же направлении. КШД не выдаёт никаких импульсов синхронизации!

def main():
	# central point
#	x_med = 9000
#	y_med = 17300
	motor_config = DeviceConfig("Support")
	mx = Motor(motor_config.ADDR[0], motor_config.PORT, motor_config.BAUDRATE, debug=True)
	my = Motor(motor_config.ADDR[1], motor_config.PORT, motor_config.BAUDRATE, debug=True)
	
	mx.get_abs_position()
	x_med = mx.position
	my.get_abs_position()
	y_med = my.position
	
	n_steps = x_zone * y_zone *2
	t = (x_step_time*1e-9 * x_zone * y_zone + y_step_time*1e-9 * y_zone + 0.65*y_zone) *2

	print("Total steps per (forward-backward) cycle:", n_steps)
	print("Total time per cycle:",  t, "sec.")

	print("Total time for", n_cycles, "cycles:",	int(t*n_cycles) // 60, "min.",
													int(t*n_cycles) % 60, "sec.")

	if "n" in input("Procceed? ('no' to cancel) "):
		sys.exit()

	if "-c" in sys.argv[0:]:
		print("Calibrating")
		dev.calibrate()

	print("Moving to the initial point")
	mx.give_pulse()
	my.give_pulse()
	mx.go_with_acc(-x_zone//2)
	mx.wait_until_ready()
	
	my.go_with_acc(-y_zone//2)
	my.wait_until_ready()
	
	mx.frequency_test()
	my.frequency_test()

	input("Press <Enter> to start scan.")

	mx.set_period(x_timer)
	mx.save_data()
	my.set_period(y_timer)
	my.save_data()

	t0 = time.time()
	try:
		for cycle in range(int(n_cycles)):
			x_dir = 1
			y_dir = 1
			for direction in [0, 1]:
				if direction==0:
					print(F"Cycle #{cycle}, forward direction.")
				else:
					print(F"Cycle #{cycle}, backward direction.")
				n_rows = 0
				while n_rows<y_zone:
					print(F"Row {n_rows} of {int(y_zone)} (cycle {cycle+0.5*direction} of {int(n_cycles)})")
					print(x_dir*x_zone, x_step_time)
					my.go_accurately(x_dir*x_zone, x_step_time)
					time.sleep(x_zone*x_step_time*1e-9*0.8)
#					mx.go_with_acc(x_dir*x_zone)
					print("started")
					sys.exit()
					my.wait_until_ready()
					print("waited")
					time.sleep(0.1)
#					my.go_accurately(y_dir, y_step_time)
					my.go_with_acc(y_dir*10)
					my.wait_until_ready()
					n_rows += 1
					x_dir *= -1
				y_dir *= -1
	except KeyboardInterrupt:
		pass
	print(time.time() - t0)

	print("Moving to the central point")
	mx.block()
	my.block()
	mx.get_abs_position()
	mx.go_with_acc(x_med - mx.position)
	my.get_abs_position()
	my.go_with_acc(y_med - my.position)
	mx.release()
	my.wait_until_ready()	
	mx.wait_until_ready()



if __name__ == "__main__":
	main()


