#!/usr/bin/env python3
#!coding=utf8
import motor, time

'''
from pygame import *
import pygame, time, numpy, pygame.sndarray
sample_rate = 44100
pygame.mixer.pre_init(sample_rate, -16, 2) # 44.1kHz, 16-bit signed, stereo
pygame.init()

def play_for(sample_array, ms, volLeft, volRight):
  sound = pygame.sndarray.make_sound(sample_array)
  beg = time.time()
  channel = sound.play(-1)
  channel.set_volume(volLeft,volRight)
  pygame.time.delay(ms)
  sound.stop()
  end = time.time()
  return beg, end
    
def sine_array_onecycle(hz, peak):
  length = sample_rate / float(hz)
  omega = numpy.pi * 2 / length
  xvalues = numpy.arange(int(length)) * omega
  return (peak * numpy.sin(xvalues))
    
def sine_array(hz, peak, n_samples = sample_rate):
  return numpy.resize(sine_array_onecycle(hz, peak), (n_samples,))

'''


d2 = 1./2
d4 = 1./4
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

tempo = 80.
flat_keys = ["H", "E"]
sharp_keys = []
notes_up = [[["G", 1, d4],  ["G", 1, d4],  ["G", 1, d4],  ["E", 1, d8p],  ["H", 1, d16]],
            [["G", 1, d4],  ["E", 1, d8p], ["H", 1, d16], ["G", 1, d2]],
            [["D", 2, d4],  ["D", 2, d4],  ["D", 2, d4],  ["E", 2, d8p],  ["H", 1, d16]],
            [["Gf", 1, d4], ["E", 1, d8p], ["H", 1, d16], ["Gn", 1, d2]],
            [["G", 2, d4],  ["G", 1, d8p], ["G", 1, d16], ["G", 2, d4],  ["Gf", 2, d8p], ["F", 1, d16]], #
            [["En", 2, d16],["Ds", 2, d16],["En", 2, d8], ["0", 0, d16], ["Gs", 1, d8],  ["Cs", 2, d4],  ["Hs", 1, d8p], ["H", 1, d16],],
            [["H", 1, d16], ["A", 1, d16], ["H", 1, d8],  ["0", 0, d16], ["E", 1, d8],   ["Gf", 1, d4],  ["E", 1, d8p],  ["G", 1, d16],],
            [["H", 1, d4],  ["G", 1, d8p], ["H", 1, d16], ["D", 2, d2],], 
            [["G", 2, d4],  ["G", 1, d8p], ["G", 1, d16], ["G", 2, d4],  ["Gf", 2, d8p], ["F", 1, d16]], #
            [["En", 2, d16],["Ds", 2, d16],["En", 2, d8], ["0", 0, d16], ["Gs", 1, d8],  ["Cs", 2, d4],  ["Hs", 1, d8p], ["H", 1, d16],],
            [["H", 1, d16], ["A", 1, d16], ["H", 1, d8],  ["0", 0, d16], ["E", 1, d8],   ["Gf", 1, d4],  ["E", 1, d8p],  ["G", 1, d16],],
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
mx = motor.Motor(3, '/dev/ttyUSB0', 57600, timeout=0.016)
mx.set_logical_address(338)
time.sleep(0.1)
my = motor.Motor(2, '/dev/ttyUSB0', 57600, timeout=0.016)
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
  for beat in notes:
    flats = []
    sharps = []
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
#      print(duration, t)
      if frequency>20.:
        seq.append([int(frequency), int(frequency*duration), note[2], t])
      else:
        seq.append([duration, note[2], t])
      t += note[2]

sgn = [+1, +1]

try:
	while seq_up and seq_down:
		note = [seq_up[0], seq_down[0]]
		mot = []
		if note[0][-1] == note[1][-1]:
		    print(note)
		    seq_up.pop(0)
		    seq_down.pop(0)
		    if len(note[0])>3:
		        mot.append([mx, 0])
		    if len(note[1])>3:
		        mot.append([my, 1])
		elif note[0][-1]<note[1][-1]:
		    print(note[0])
		    seq_up.pop(0)
		    if len(note[0])>3:
		        mot.append([mx, 0])
		else:
		    print(note[1])
		    seq_down.pop(0)
		    if len(note[1])>3:
		        mot.append([my, 1])
		mx.block()
		my.block()
		for m, n_mot in mot:
		    m.wait_until_ready()
		    m.set_device_speed(int(note[n_mot][0]), 10000, 1000)
		    m.go_wo_acc(sgn[n_mot] * note[n_mot][1])
		    sgn[n_mot] *= -1
		mx.release()
		time.sleep(min(note[0][-2], note[1][-2])*4*60./tempo)


except KeyboardInterrupt:
	print('Stopped\n')
mx.get_abs_position()
mx.go_with_acc(init_x_position-mx.position)
my.get_abs_position()
my.go_with_acc(init_y_position-my.position)

mx.wait_until_ready()
my.wait_until_ready()

import set_logical_address

