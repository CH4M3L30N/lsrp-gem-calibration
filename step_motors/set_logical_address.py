#!/usr/bin/env python3
from motor import Motor
import time

PORT = "/dev/ttyUSB0"
#BAUDRATE = 57600
BAUDRATE = 19200
                                     # addr, sn, MoveI, StopI, K+, K-, K0, V_min, V_max, Acc
config = (  {"addr" : 2, "sn" : 345, "i_move" : 1.0, "i_stop" : 1.0, "t_stop" : 20,
             "K+" : 1, "K-" : 1, "K0" : 0, "v_min" : 140, "v_max" : 2000, "acc" : 10000}, # мотор Y
            {"addr" : 3, "sn" : 338, "i_move" : 1.0, "i_stop" : 0.2, "t_stop" : 20,
             "K+" : 1, "K-" : 1, "K0" : 0, "v_min" : 140, "v_max" : 3000, "acc" : 10000}, ) # мотор Y

I = {0.:0, 0.2:1, 0.3:2, 0.5:3, 0.6:4, 1.0:5, 2.0:6, 3.5:7}

CFG = 0b1

for c in config:
    m = Motor(addr=c["addr"], ioport=PORT, baudrate=BAUDRATE)
    m.immediate_stop()
    m.set_logical_address(c["sn"])
    time.sleep(0.1)
    print(m.get_device_info())
#    m.get_device_status()
    print((m.get_device_config()["CFG"]))
    m.set_device_config(i_move=I[c["i_move"]], i_stop=I[c["i_stop"]], t_stop=c["t_stop"],
                        cfg=CFG|(c["K+"]<<3)|(c["K-"]<<2)|(c["K0"]<<4))
    m.set_device_speed(c["v_min"], c["v_max"], c["acc"])
