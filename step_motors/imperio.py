#!/usr/bin/env python3
#!coding=utf8
import time
from motor import Motor

'''
import pygame
import numpy
from math import pi
sample_rate = 44100
pygame.mixer.pre_init(sample_rate, -16, 1) # 44.1kHz, 16-bit signed, stereo
pygame.init()

block = False

class Motor:
    def __init__(self, addr, port, baudrate, timeout=0.016):
        self.v = 1000
        self.position = 0
    
    def block(self):
        global block
        block = True
    
    def release(self):
        global block
        block = False
        if hasattr(self, "sound"):
            self.channel = self.sound.play(abs(self.steps))
            self.channel.set_volume(1.0)
    
    def set_device_speed(self, v_min, v_max, acc):
        self.v = int(v_min)
        self.onecycle = numpy.array((2**14)*numpy.sin(2*pi*self.v * numpy.arange(sample_rate/self.v)/sample_rate)**2, dtype="i2")
        
    def go_wo_acc(self, steps):
        self.sound = pygame.sndarray.make_sound(self.onecycle)
        self.steps = steps
        if not block:
            self.channel = self.sound.play(abs(self.steps))
            self.channel.set_volume(1.0)

    def go_w_acc(self, steps):
        pass
        
    def wait_until_ready(self):
        while pygame.mixer.get_busy():
            time.sleep(0.01)
    
    def set_logical_address(self, addr):
        pass
        
    def get_abs_position(self):
        pass
    
#'''


#"""
d1 = 1.
d2 = 1./2
d2p = 1./2 + 1./4
d4 = 1./4
d4p = 1./4 + 1./8
d8 = 1./8
d8p = 1./8 + 1./16
d16 = 1./16
# f (flat) - бемоль
# n (natural) - бекар
# s (sharp) - диез
freq = {"A" : 440.}
freq["H"] = freq["A"]*2**( 2./12)
freq["G"] = freq["A"]*2**(-2./12)
freq["F"] = freq["G"]*2**(-2./12)
freq["E"] = freq["F"]*2**(-1./12)
freq["D"] = freq["E"]*2**(-2./12)
freq["C"] = freq["D"]*2**(-2./12)
freq["0"] = 0
for n in ["A", "H", "C", "D", "E", "F", "G"]:
  freq[n + "s"] = freq[n] * 2**(+1./12)
  freq[n + "f"] = freq[n] * 2**(-1./12)
  freq[n + "n"] = freq[n]

#'''
tempo = 80.
flat_keys = ["H", "E"]
sharp_keys = []
notes_up = [[["G", 1, d4],  ["G", 1, d4],  ["G", 1, d4],  ["E", 1, d8p],  ["H", 1, d16]],
            [["G", 1, d4],  ["E", 1, d8p], ["H", 1, d16], ["G", 1, d2]],
            [["D", 2, d4],  ["D", 2, d4],  ["D", 2, d4],  ["E", 2, d8p],  ["H", 1, d16]],
            [["Gf", 1, d4], ["E", 1, d8p], ["H", 1, d16], ["Gn", 1, d2]],
            [["G", 2, d4],  ["G", 1, d8p], ["G", 1, d16], ["G", 2, d4],  ["Gf", 2, d8p], ["F", 1, d16]], #
            [["En", 2, d16],["Ds", 2, d16],["En", 2, d8], ["0", 0, d8], ["Gs", 1, d8],  ["Cs", 2, d4],  ["Hs", 1, d8p], ["H", 1, d16],],
            [["H", 1, d16], ["A", 1, d16], ["H", 1, d8],  ["0", 0, d8], ["E", 1, d8],   ["Gf", 1, d4],  ["E", 1, d8p],  ["G", 1, d16],],
            [["H", 1, d4],  ["G", 1, d8p], ["H", 1, d16], ["D", 2, d2],], 
            [["G", 2, d4],  ["G", 1, d8p], ["G", 1, d16], ["G", 2, d4],  ["Gf", 2, d8p], ["F", 1, d16]], #
            [["En", 2, d16],["Ds", 2, d16],["En", 2, d8], ["0", 0, d8], ["Gs", 1, d8],  ["Cs", 2, d4],  ["Hs", 1, d8p], ["H", 1, d16],],
            [["H", 1, d16], ["A", 1, d16], ["H", 1, d8],  ["0", 0, d8], ["E", 1, d8],   ["Gf", 1, d4],  ["E", 1, d8p],  ["G", 1, d16],],
            [["G", 1, d4],  ["E", 1, d8p], ["H", 1, d16], ["G", 1, d2],]
]

notes_down = [[["G", -1, d4],  ["G", -1, d4], ["G", -1, d4], ["E", -1, d4]],
              [["G", -1, d4],  ["E", -1, d4], ["G", -1, d4], ["G", -1, d4]],
              [["G", -1, d4],  ["G", -1, d4], ["G", -1, d4], ["E", -1, d4]],
              [["E", -1, d4],  ["E", -1, d4], ["G", -1, d4], ["G", -1, d4]],
              [["G", -1, d4],  ["G", -1, d4], ["G", -1, d4], ["G", -1, d4]],
              [["Df", -1, d4],  ["D", -1, d4], ["D", -1, d4], ["D", -1, d4]],
              [["E", -1, d4],  ["E", -1, d4], ["E", -1, d4], ["E", -1, d4]],
              [["G", -1, d4],  ["E", -1, d4], ["G", -1, d4], ["G", -1, d4]],
              [["E", -1, d4],  ["E", -1, d4], ["G", -1, d4], ["G", -1, d4]],
              [["Df", -1, d4],  ["Df", 0, d4], ["D", -1, d4], ["D", 0, d4]],
              [["E", -1, d4],  ["E", 0, d4], ["E", -1, d4], ["E", 0, d4]],
              [["G", -1, d4],  ["C", -1, d4], ["G", -1, d4], ["G", -1, d4]],
]
#'''

'''
tempo = 76.
flat_keys = []
sharp_keys = []
notes_up = [[["0", 1, d2], ["0", 1, d4],  ["0", 1, d8],  ["G", 1, d8]],
            [["C", 2, d4], ["G", 1, d8p], ["A", 1, d16], ["H", 1, d4], ["E", 1, d8],  ["E", 1, d8]],
            [["A", 1, d4], ["G", 1, d8p], ["F", 1, d16], ["G", 1, d4], ["C", 1, d8], ["C", 1, d8]],
            [["D", 1, d4], ["D", 1, d8p], ["E", 1, d16], ["F", 1, d4], ["F", 1, d8p], ["G", 1, d16]],
            [["A", 1, d4], ["H", 1, d8], ["C", 2, d8], ["D", 2, d4p], ["G", 1, d8]],
            [["C", 2, d4], ["G", 1, d8p], ["A", 1, d16], ["H", 1, d4], ["H", 1, d8], ["G", 1, d8]],
            [["A", 1, d4], ["H", 1, d8p], ["A", 1, d16], ["H", 1, d4], ["E", 1, d8], ["E", 1, d8]],
            [["A", 1, d4], ["G", 1, d8p], ["F", 1, d16], ["G", 1, d4], ["C", 1, d8p], ["C", 1, d16]],
            [["D", 1, d4], ["E", 1, d8p], ["Fs", 1, d16], ["G", 1, d2]],
]

notes_down = [[["C", -1, d2p],  ["0", -1, d4]],
              [["C", -1, d4], ["C", -1, d4], ["E", -1, d4], ["E", -1, d4]],
              [["F", -1, d4], ["F", -1, d4], ["E", -1, d4], ["E", -1, d4]],
              [["D", -1, d4], ["D", -1, d4], ["C", -1, d2]],
              [["C", -1, d8], ["C", -1, d8], ["H", -2, d8], ["A", -2, d8], ["G", -2, d8], ["F", -1, d8], ["E", -1, d8], ["D", -1, d8]],
              [["C", -1, d4], ["E", -1, d4], ["G", -1, d4p], ["G", -1, d8]],
              [["A", -2, d4], ["C", -1, d4], ["E", -1, d4p], ["E", -1, d8]],
              [["F", -1, d4], ["D", -1, d4], ["E", -1, d4], ["E", -1, d4]],
              [["D", -1, d4], ["D", -1, d4], ["G", -1, d8], ["F", -1, d8], ["E", -1, d8], ["D", -1, d8]], 
]
#'''


mx = Motor(3, '/dev/ttyUSB0', 57600, timeout=0.016)
mx.set_logical_address(338)
time.sleep(0.1)
my = Motor(2, '/dev/ttyUSB0', 57600, timeout=0.016)
my.set_logical_address(345)
time.sleep(0.1)

mx.get_abs_position()
init_x_position = mx.position
time.sleep(0.1)

my.get_abs_position()
init_y_position = my.position
#'''

seq_up = []
seq_down = []


for seq, notes in [[seq_up, notes_up], [seq_down, notes_down]]:
  t = 0
  print("!!!")
  for beat in notes:
    flats = []
    sharps = []
    count = 0
    for note in beat:
#      print(note, end="\t")
      frequency = freq[note[0]]
      # ключевые знаки
      if note[0] in sharp_keys:
        frequency *= 2**(+1./12)
      if note[0] in flat_keys:
        frequency *= 2**(-1./12)
      # случайные знаки
      if note[0] in sharps:
        frequency *= 2**(1./12)
      if note[0] in flats:
        frequency *= 2**(-1./12)
      if "s" in note[0]:
        sharps.append(note[0][0])
      if "f" in note[0]:
        flats.append(note[0][0])
      # бекары уже есть
      frequency *= 2**(note[1] - 1)
      duration = note[2]*4*60./tempo
      count += duration
#      print(duration, t)
      if frequency>20.:
        seq.append({"speed" : int(frequency), "dur" : duration, "steps" : int(frequency*duration), "time" : t})
      else:
        seq.append({"dur" : duration, "time" : t})
      t += note[2]
    print(count)
sgn_x = +1
sgn_y = +1

#'''
try:
    while seq_up: # and seq_down:
        note_up = seq_up[0]
        note_down = seq_down[0]
        mot = []
        print(note_up, note_down)
        if note_up["time"] == note_down["time"]:
            print("move x, move y")
            mx.wait_until_ready()
            my.wait_until_ready()
            mx.block()
            my.block()
            if "speed" in note_up:
                mx.set_device_speed(note_up["speed"], 10000, 1000)
                mx.go_wo_acc(sgn_x * note_up["steps"])
                sgn_x *= -1
            if "speed" in note_down:
                my.set_device_speed(note_down["speed"], 10000, 1000)
                my.go_wo_acc(sgn_y * note_down["steps"])
                sgn_y *= -1
            mx.release()
#            my.release()
            time.sleep(min(note_up["dur"], note_down["dur"]))
            seq_up.pop(0)
            seq_down.pop(0)
        elif note_up["time"] < note_down["time"]:
            mx.wait_until_ready()
            print("move x")
            if "speed" in note_up:
                mx.set_device_speed(note_up["speed"], 10000, 1000)
                mx.go_wo_acc(sgn_x * note_up["steps"])
                sgn_x *= -1
            seq_up.pop(0)
        elif note_up["time"] > note_down["time"]:
            my.wait_until_ready()
            print("move y")
            if "speed" in note_down:
                my.set_device_speed(note_down["speed"], 10000, 1000)
                my.go_wo_acc(sgn_y * note_down["steps"])
                sgn_y *= -1
            seq_down.pop(0)
    mx.wait_until_ready()
    my.wait_until_ready()   
    
except KeyboardInterrupt:
    print('Stopped\n')
time.sleep(3.0)

import set_logical_address
time.sleep(1)

mx.get_abs_position()
mx.go_with_acc(init_x_position-mx.position)
my.get_abs_position()
my.go_with_acc(init_y_position-my.position)

mx.wait_until_ready()
my.wait_until_ready()

#'''
#"""
